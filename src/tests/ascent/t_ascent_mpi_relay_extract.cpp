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
/// file: ascent_mpi_render_2d.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_relay_extract_iso)
{

    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 100;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_hd5f_iso");
    // remove old files before rendering
    remove_test_image(output_file);
    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "radial_vert";
    contour_params["iso_values"] = 250.;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";

    extracts["e1/params/path"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    if(par_rank == 0)
    {
      std::string msg = "An example of using an relay extract to save the results of "
                        " a pipeline to the file system.";
      ASCENT_ACTIONS_DUMP(actions,output_file,msg);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_relay_extract_selected_fields)
{

    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_hd5f_sub_mesh");
    // remove old files before rendering
    remove_test_image(output_file);
    //
    // Create the actions.
    //


    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/fields"].append() = "radial_vert";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_relay_extract_mesh)
{

    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 101;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_hd5f_mesh");
    // remove old files before rendering
    remove_test_image(output_file);
    //
    // Create the actions.
    //


    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    if(par_rank == 0)
    {
      std::string msg = "An example of using an relay extract to save the published mesh "
                        "to the file system.";
      ASCENT_ACTIONS_DUMP(actions,output_file,msg);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_relay_partially_empty)
{

    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");
    //
    // Create the data.
    //
    Node data, data2, verify_info;
    create_3d_example_dataset(data,32,par_rank,par_size*2);
    create_3d_example_dataset(data2,32,par_rank+1,par_size*2);
    data["state/cycle"] = 101;
    data2["state/cycle"] = 101;

    Node multi_dom;
    // make a multi domain data set
    conduit::Node &dom1 = multi_dom.append();
    dom1.set_external(data);
    conduit::Node &dom2 = multi_dom.append();
    dom2.set_external(data2);

    float64_array field = data["fields/radial_vert/values"].value();
    int size = field.number_of_elements();
    float64 val = 1.;

    if(par_rank == 0)
    {
      val = 0.;
    }

    for(int i = 0; i < size; ++i)
    {
      field[i] = val;
    }

    float64_array field2 = data2["fields/radial_vert/values"].value();
    int size2 = field2.number_of_elements();
    for(int i = 0; i < size2; ++i)
    {
      field2[i] = 1.0;
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_hd5f_mesh");
    // remove old files before rendering
    remove_test_image(output_file);
    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "threshold";
    // filter knobs
    conduit::Node &thresh_params = pipelines["pl1/f1/params"];
    thresh_params["field"] = "radial_vert";
    thresh_params["min_value"] = 0.9;
    thresh_params["max_value"] = 1.1;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = output_file;

    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    //ascent.publish(data);
    ascent.publish(multi_dom);
    ascent.execute(actions);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_runtime, test_relay_empty)
{

    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");
    //
    // Create the data.
    //
    Node data, verify_info;
    create_3d_example_dataset(data,32,par_rank,par_size);
    data["state/cycle"] = 101;

    Node multi_dom;
    // make a multi domain data set
    conduit::Node &dom1 = multi_dom.append();
    dom1.set_external(data);

    float64_array field = data["fields/radial_vert/values"].value();
    int size = field.number_of_elements();
    float64 val = 1.;

    if(par_rank == 0)
    {
      val = 0.;
    }

    for(int i = 0; i < size; ++i)
    {
      field[i] = val;
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    // make sure the _output dir exists
    string output_path = "";
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_file = conduit::utils::join_file_path(output_path,"tout_hd5f_mesh");
    // remove old files before rendering
    remove_test_image(output_file);
    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "threshold";
    // filter knobs
    conduit::Node &thresh_params = pipelines["pl1/f1/params"];
    thresh_params["field"] = "radial_vert";
    thresh_params["min_value"] = 0.9;
    thresh_params["max_value"] = 1.1;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = output_file;

    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    //ascent.publish(data);
    ascent.publish(multi_dom);
    ascent.execute(actions);
    ascent.close();

}

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_bp_num_files)
{
    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");

    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);

    // rank 0 gets first 4 domains, rank 1 gets the rest
    if(par_rank == 0)
    {
        data.remove(4);
        data.remove(4);
        data.remove(4);
    }
    else if(par_rank == 1)
    {
        data.remove(0);
        data.remove(0);
        data.remove(0);
        data.remove(0);
    }
    else
    {
        // cyrus was wrong about 2 mpi ranks.
        EXPECT_TRUE(false);
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract num_files option with mpi");

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    // lets try with -1 to 8 files.

    // nfiles less than 1 should trigger default case
    // (n output files = n domains)
    for(int nfiles=-1; nfiles < 9; nfiles++)
    {
        std::cout << "[" << par_rank <<  "] test nfiles = " << nfiles << std::endl;
        MPI_Barrier(comm);
        oss.str("");
        oss << "tout_relay_mpi_extract_nfiles_" << nfiles;

        string output_base = conduit::utils::join_file_path(output_path,
                                                            oss.str());

        string output_dir  = output_base + ".cycle_000000";
        string output_root = output_base + ".cycle_000000.root";

        if(par_rank == 0)
        {
            // remove existing directory
            utils::remove_directory(output_dir);
            utils::remove_directory(output_root);
        }

        MPI_Barrier(comm);

        conduit::Node actions;
        // add the extracts
        conduit::Node &add_extracts = actions.append();
        add_extracts["action"] = "add_extracts";
        conduit::Node &extracts = add_extracts["extracts"];

        extracts["e1/type"]  = "relay";
        extracts["e1/params/path"] = output_base;
        extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
        extracts["e1/params/num_files"] =  nfiles;

        //
        // Run Ascent
        //

        Ascent ascent;

        Node ascent_opts;
        // we use the mpi handle provided by the fortran interface
        // since it is simply an integer
        ascent_opts["runtime"] = "ascent";
        ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
        ascent.open(ascent_opts);
        ascent.publish(data);
        ascent.execute(actions);
        ascent.close();

        MPI_Barrier(comm);

        // count the files
        //  file_%06llu.{protocol}:/domain_%06llu/...
        int nfiles_to_check = nfiles;
        if(nfiles <=0 || nfiles == 8) // expect 7 files (one per domain)
        {
            nfiles_to_check = 7;
        }

        EXPECT_TRUE(conduit::utils::is_directory(output_dir));
        EXPECT_TRUE(conduit::utils::is_file(output_root));

        char fmt_buff[64] = {0};
        for(int i=0;i<nfiles_to_check;i++)
        {

            std::string fprefix = "file_";
            if(nfiles_to_check == 7)
            {
                // in the n domains == n files case, the file prefix is
                // domain_
                fprefix = "domain_";
            }
            snprintf(fmt_buff, sizeof(fmt_buff), "%06d",i);
            oss.str("");
            oss << conduit::utils::join_file_path(output_base + ".cycle_000000",
                                                  fprefix)
                << fmt_buff << ".hdf5";
            std::string fcheck = oss.str();
            std::cout << " checking: " << fcheck << std::endl;
            EXPECT_TRUE(conduit::utils::is_file(fcheck));
        }

        MPI_Barrier(comm);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_mpi_sparse_topos_1)
{
    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");

    Node data;
    ostringstream oss;

    // create mesh where each rank has three domains with different topos
    for(index_t d =0; d<3; d++)
    {
        Node &mesh = data.append();

        mesh["state/cycle"] = 0;

        oss.str("");
        oss << "my_coords_rank_" <<  par_rank << "_" << d;
        std::string c_name = oss.str();

        oss.str("");
        oss << "my_topo_rank_" <<  par_rank << "_" << d;
        std::string t_name = oss.str();

        oss.str("");
        oss << "my_field_rank_" <<  par_rank << "_" << d;
        std::string f_name = oss.str();

        // create the coordinate set
        mesh["coordsets"][c_name]["type"] = "uniform";
        mesh["coordsets"][c_name]["dims/i"] = 3;
        mesh["coordsets"][c_name]["dims/j"] = 3;
        mesh["coordsets"][c_name]["origin/x"] = -10.0;
        mesh["coordsets"][c_name]["origin/y"] = -10.0;
        mesh["coordsets"][c_name]["spacing/dx"] = 10.0;
        mesh["coordsets"][c_name]["spacing/dy"] = 10.0;

        mesh["topologies"][t_name]["type"] = "uniform";
        mesh["topologies"][t_name]["coordset"] = c_name;

        mesh["fields"][f_name]["association"] =  "element";
        mesh["fields"][f_name]["topology"] =  t_name;
        mesh["fields"][f_name]["values"].set(DataType::float64(4));

        float64 *ele_vals_ptr = mesh["fields"][f_name]["values"].value();

        for(int i=0;i<4;i++)
        {
            ele_vals_ptr[i] = float64(d);
        }
    }

    Node verify_info;
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing relay extract with sparse topos in parallel");

    std::string output_path;

    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }


    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_mpi_sparse_topos_1");

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";

    Ascent ascent;
    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);
    
    // read back in the blueprint index and make sure it 
    // has what we expect in it

    Node n_root;
    conduit::relay::io::load(output_file + ".cycle_000000.root","hdf5",n_root);

    const Node &bp_index = n_root["blueprint_index/mesh"];

    EXPECT_EQ(bp_index["state/number_of_domains"].to_int(),6);
    EXPECT_EQ(bp_index["coordsets"].number_of_children(),6);
    EXPECT_EQ(bp_index["topologies"].number_of_children(),6);
    EXPECT_EQ(bp_index["fields"].number_of_children(),6);
    
}


//-----------------------------------------------------------------------------
TEST(ascent_relay, test_relay_mpi_sparse_topos_2)
{
    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    ASCENT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");

    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //

    Node data;
    ostringstream oss;
    
    // rank 1 have 3 domains, rank zero none
    if(par_rank > 0)
    {

        // three domains with different topos
        for(index_t d =0; d<3; d++)
        {
            Node &mesh = data.append();

            mesh["state/cycle"] = 0;

            oss.str("");
            oss << "my_coords_rank_" <<  par_rank << "_" << d;
            std::string c_name = oss.str();

            oss.str("");
            oss << "my_topo_rank_" <<  par_rank << "_" << d;
            std::string t_name = oss.str();

            oss.str("");
            oss << "my_field_rank_" <<  par_rank << "_" << d;
            std::string f_name = oss.str();

            // create the coordinate set
            mesh["coordsets"][c_name]["type"] = "uniform";
            mesh["coordsets"][c_name]["dims/i"] = 3;
            mesh["coordsets"][c_name]["dims/j"] = 3;
            // add origin and spacing to the coordset (optional)
            mesh["coordsets"][c_name]["origin/x"] = -10.0;
            mesh["coordsets"][c_name]["origin/y"] = -10.0;
            mesh["coordsets"][c_name]["spacing/dx"] = 10.0;
            mesh["coordsets"][c_name]["spacing/dy"] = 10.0;

            // add the topology
            // this case is simple b/c it's implicitly derived from the coordinate set
            mesh["topologies"][t_name]["type"] = "uniform";
            // reference the coordinate set by name
            mesh["topologies"][t_name]["coordset"] = c_name;

            // add a simple element-associated field
            mesh["fields"][f_name]["association"] =  "element";
            // reference the topology this field is defined on by name
            mesh["fields"][f_name]["topology"] =  t_name;
            // set the field values, for this case we have 4 elements
            mesh["fields"][f_name]["values"].set(DataType::float64(4));

            float64 *ele_vals_ptr = mesh["fields"][f_name]["values"].value();

            for(int i=0;i<4;i++)
            {
                ele_vals_ptr[i] = float64(d);
            }
        }

        Node verify_info;
        EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    }



    ASCENT_INFO("Testing relay extract with sparse topos in parallel");

    std::string output_path;

    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }


    string output_file = conduit::utils::join_file_path(output_path,"tout_relay_mpi_sparse_topos_2");

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    conduit::Node &execute  = actions.append();
    execute["action"] = "execute";
    //
    Ascent ascent;
    //
    Node ascent_opts;
    // we use the mpi handle provided by the fortran interface
    // since it is simply an integer
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();


    MPI_Barrier(comm);

    // read back in the blueprint index and make sure it 
    // has what we expect in it

    Node n_root;
    conduit::relay::io::load(output_file + ".cycle_000000.root","hdf5",n_root);

    const Node &bp_index = n_root["blueprint_index/mesh"];

    EXPECT_EQ(bp_index["state/number_of_domains"].to_int(),3);
    EXPECT_EQ(bp_index["coordsets"].number_of_children(),3);
    EXPECT_EQ(bp_index["topologies"].number_of_children(),3);
    EXPECT_EQ(bp_index["fields"].number_of_children(),3);

}

//
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


