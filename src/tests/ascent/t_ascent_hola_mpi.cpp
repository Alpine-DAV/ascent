//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_hola_mpi.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>
#include <ascent_hola.hpp>
#include <ascent_hola_mpi.hpp>
#include <mpi.h>
#include "conduit_relay.hpp"
#include "conduit_relay_mpi.hpp"

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;
using namespace ascent;


const int CELL_DIMS = 32;

void hola_mpi_helpers_test_setup_src_data(int rank, Node &data)
{

//     src_doms[0] = 5;
//     src_doms[1] = 5;
//     src_doms[2] = 5;
//     src_doms[3] = 5;
//     src_doms[4] = 3;

    int num_local_doms = 5;
    if(rank == 4)
        num_local_doms = 3;

    for(int i=0; i < num_local_doms; i++)
    {
        Node &payload = data.append();
        payload["src_rank"] = rank;
        payload["src_local_domain_id"] = i;
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_hola_mpi, test_hola_mpi_helpers)
{
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = relay::mpi::rank(comm);
    int total_size = relay::mpi::size(comm);
    int rank_split = 5;
    // calc source size using split
    // source ranks are 0 to rank_split - 1
    int src_size = rank_split;

    // calc dest size using split
    // dest ranks are rank_split  to total_size -1
    int dest_size = total_size - rank_split;

    Node data;
    if(rank < src_size)
    {
        hola_mpi_helpers_test_setup_src_data(rank,data);
    }
    else
    {
        data.reset();
    }

    for(int i=0;i<total_size;i++)
    {
        if(rank == i)
        {
            std::cout << "rank " << i << ": " << std::endl;
            data.print();
        }
        MPI_Barrier(comm);
    }

    //
    // create maps that map world mpi ranks to send and recv ranks
    //

    Node my_maps;
    my_maps["wts"] = DataType::int32(total_size);
    my_maps["wtd"] = DataType::int32(total_size);

    int32_array world_to_src  = my_maps["wts"].value();
    int32_array world_to_dest = my_maps["wtd"].value();

    for(int i=0;i<total_size;i++)
    {
        if(i < src_size)
        {
            world_to_dest[i] = -1;
            world_to_src[i]  = i;
        }
        else
        {
            world_to_dest[i] = i - src_size;
            world_to_src[i] = -1;
        }
    }

    Node comm_map;

    hola_mpi_comm_map(data,
                      comm,
                      world_to_src,
                      world_to_dest,
                      comm_map);

    if(rank == 0)
    {
        std::cout << "Hola MPI Comm Map:" << std::endl;
        comm_map.print();
    }

    MPI_Barrier(comm);

    if(rank == 5)
        EXPECT_EQ(data.number_of_children(),0);
    if(rank == 6)
        EXPECT_EQ(data.number_of_children(),0);
    if(rank == 7)
        EXPECT_EQ(data.number_of_children(),0);

    if(rank < src_size)
    {
        int src_idx = rank;
        hola_mpi_send(data,comm,src_idx,comm_map);
    }
    else
    {
        int dest_idx = rank - rank_split;
        hola_mpi_recv(comm,dest_idx,comm_map,data);
    }

    for(int i=0;i<total_size;i++)
    {
        if(rank == i)
        {
            std::cout << "rank " << i << ": " << std::endl;
            data.print();
        }
        MPI_Barrier(comm);
    }

    if(rank == 5)
        EXPECT_EQ(data.number_of_children(),7);
    if(rank == 6)
        EXPECT_EQ(data.number_of_children(),7);
    if(rank == 7)
        EXPECT_EQ(data.number_of_children(),9);
}

//-----------------------------------------------------------------------------
TEST(ascent_hola_mpi, test_hola_mpi)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
        return;
    }

    //
    // Set Up MPI
    //
    int world_rank;
    int world_size;
    MPI_Comm world_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int color = 0;
    int rank_split = 5;

    if(world_rank > 4)
    {
        color = 1;
    }

    MPI_Comm sub_comm;
    MPI_Comm_split(world_comm,color,world_rank,&sub_comm);

    int sub_rank =0;
    int sub_size =0;

    MPI_Comm_rank(sub_comm, &sub_rank);
    MPI_Comm_size(sub_comm, &sub_size);

    for(int i=0;i<world_size;i++)
    {
        if(world_rank == i)
        {
        std::cout << "world_rank " << world_rank
                  << " world_comm_size " << world_rank
                  << " sub_rank " << sub_rank
                  << " sub_comm_size " << sub_size << std::endl;
        }
        MPI_Barrier(world_comm);
    }

    if(color == 0)
    {

        //
        // Create example data
        //
        Node data, verify_info;
        create_3d_example_dataset(data, CELL_DIMS, world_rank,5);

        // hi-jack the radial_vert field and override it with rank
        float64_array vals = data["fields/radial_vert/values"].value();
        for(int i=0;i< vals.number_of_elements();i++)
        {
            vals[i] = world_rank;
        }

        EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

        int cycle = 100;
        data["state/cycle"] = cycle;


        //
        // Create the actions to export the dataset
        //

        conduit::Node actions;
        // add the extract
        conduit::Node &add_extract = actions.append();
        add_extract["action"] = "add_extracts";
        add_extract["extracts/e1/type"]  = "hola_mpi";
        add_extract["extracts/e1/params/mpi_comm"] = MPI_Comm_c2f(world_comm);
        add_extract["extracts/e1/params/rank_split"] = rank_split;
        actions.append()["action"] = "execute";
        //
        // Run Ascent
        //
        Ascent ascent;
        Node ascent_opts;
        ascent_opts["mpi_comm"] = MPI_Comm_c2f(sub_comm);
        ascent.open(ascent_opts);
        ascent.publish(data);
        ascent.execute(actions);
        ascent.close();

        //have all ranks check the output file
        MPI_Barrier(world_comm);

        string output_image = conduit::utils::join_file_path(output_dir(),
                                                            "tout_hola_mpi_test_render");
        EXPECT_TRUE(utils::is_file(output_image + "100.png"));
    }
    else
    {
        // use hola to say hello to the extract data
        Node data, hola_opts, actions, verify_info;
        hola_opts["mpi_comm"]   = MPI_Comm_c2f(world_comm);
        hola_opts["rank_split"] = rank_split;
        ascent::hola("hola_mpi", hola_opts, data);


        EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

        string output_image = conduit::utils::join_file_path(output_dir(),
                                                             "tout_hola_mpi_test_render");


        if(sub_rank == 0)
        {
            // make sure output dir exists
            prepare_output_dir();
            // remove old images before rendering
            remove_test_image(output_image);
        }

        Ascent ascent;
        Node ascent_opts;
        //ascent_opts["messages"] = "verbose";
        ascent_opts["mpi_comm"] = MPI_Comm_c2f(sub_comm);
        ascent.open(ascent_opts);
        //
        // render the result from hola
        //
        conduit::Node &add_scene = actions.append();
        add_scene["action"] = "add_scenes";
        add_scene["scenes/scene1/plots/plt1/type"]         = "pseudocolor";
        add_scene["scenes/scene1/plots/plt1/field"] = "radial_vert";
        add_scene["scenes/scene1/image_prefix"] = output_image;

        conduit::Node &execute  = actions.append();
        execute["action"] = "execute";

        ascent.publish(data);
        ascent.execute(actions);
        ascent.close();

        //have all ranks check the output file
        MPI_Barrier(world_comm);
        EXPECT_TRUE(utils::is_file(output_image + "100.png"));
    }

    MPI_Comm_free(&sub_comm);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
