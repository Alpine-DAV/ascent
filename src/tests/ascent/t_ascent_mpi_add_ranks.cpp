//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_mpi_add_ranks.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mpi_mesh_examples.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


int NUM_DOMAINS = 8;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_add_mpi_ranks, test_mpi_add_mpi_ranks)
{
    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mpi::mesh::examples::spiral_round_robin(NUM_DOMAINS,data,comm);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//    data["state/cycle"] = 100;
//    data.print();

    ASCENT_INFO("Testing adding mpi ranks to conduit::blueprint spiral input\n");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_mpi_add_ranks");
    string image_file = conduit::utils::join_file_path(output_path,"tout_mpi_add_ranks10");

    // remove old images before rendering
    if(par_rank == 0)
      remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];

    // pipeline 1
    pipelines["pl1/f1/type"] = "add_mpi_ranks";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["topology"] = "topo";
    params["output"] = "rank"; 

    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "rank";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/plots/p1/color_table/discrete"] = "true";
    scenes["s1/image_prefix"] = image_file;

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    if(par_rank == 0)
    {
      EXPECT_TRUE(check_test_image(output_file));
      std::string msg = "An example of adding mpi ranks to the data.";
      ASCENT_ACTIONS_DUMP(actions,output_file,msg);
    }
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


