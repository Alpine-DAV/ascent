//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_partition.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint_mpi_mesh_examples.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;

int NUM_DOMAINS = 8;

//-----------------------------------------------------------------------------
TEST(ascent_partition, test_mpi_partition_target_1)
{
    Node n;
    ascent::about(n);

    // only run this test if ascent was built with hdf5 +  vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled" ||
       n["runtimes/ascent/hdf5/status"].as_string() == "disabled" )
    {
        ASCENT_INFO("Ascent Rendering and/or HDF5 support are disabled, skipping test");
        return;
    }

    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mpi::mesh::examples::spiral_round_robin(NUM_DOMAINS,data,comm);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing blueprint partition of multi-domain mesh with MPI");

    string output_path;
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_base = conduit::utils::join_file_path(output_path,
                                                        "tout_partition_target_1_mpi");
    string output_root = output_base + "_result.cycle_000000.root";

    // remove existing file
    if(utils::is_file(output_root))
    {
        utils::remove_file(output_root);
    }

    conduit::Node actions;
    int target = 1;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"]  = "partition";
    pipelines["pl1/f1/params/target"] = target;

    //add the extract
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_base + "_result";
    extracts["e1/params/protocol"] = "hdf5";

    extracts["einput/type"] = "relay";
    extracts["einput/params/path"] = output_base + "_input";
    extracts["einput/params/protocol"] = "hdf5";

    // Add a scene that shows domain id
    //
    //add the scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "rank";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = conduit::utils::join_file_path(output_path, "tout_mpi_partition_target_1_result_render");

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);

    if(par_rank == 0)
    {
      EXPECT_TRUE(conduit::utils::is_file(output_root));
      Node read_mesh;
      conduit::relay::io::blueprint::load_mesh(output_root,read_mesh);

      int num_doms = conduit::blueprint::mesh::number_of_domains(read_mesh);
      EXPECT_TRUE(num_doms == target);
    }
}

// ----------------------------------------------------------
TEST(ascent_partition, test_mpi_partition_target_10)
{
    Node n;
    ascent::about(n);

    // only run this test if ascent was built with hdf5 +  vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled" ||
       n["runtimes/ascent/hdf5/status"].as_string() == "disabled" )
    {
        ASCENT_INFO("Ascent Rendering and/or HDF5 support are disabled, skipping test");
        return;
    }

    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mpi::mesh::examples::spiral_round_robin(NUM_DOMAINS,data,comm);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing blueprint partition of multi-domain mesh with MPI");
    
    string output_path;
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_base = conduit::utils::join_file_path(output_path,
                                                        "tout_partition_target_10_mpi");
    string output_root = output_base + "_result.cycle_000000.root";

    if(par_rank == 0)
    {
        // remove existing file
        if(utils::is_file(output_root))
        {
            utils::remove_file(output_root);
        }
    }

    conduit::Node actions;
    int target = 10;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"]  = "partition";
    pipelines["pl1/f1/params/target"] = target;

    //add the extract
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_base + "_result";
    extracts["e1/params/protocol"] = "hdf5";

    extracts["einput/type"] = "relay";
    extracts["einput/params/path"] = output_base + "_input";
    extracts["einput/params/protocol"] = "hdf5";

    // Add a scene that shows domain id
    //
    //add the scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "rank";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = conduit::utils::join_file_path(output_path, "tout_mpi_partition_target_10_result_render");

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

    MPI_Barrier(comm);

    if(par_rank == 0)
    {
      EXPECT_TRUE(conduit::utils::is_file(output_root));
      Node read_mesh;
      conduit::relay::io::blueprint::load_mesh(output_root,read_mesh);

      int num_doms = conduit::blueprint::mesh::number_of_domains(read_mesh);
      EXPECT_TRUE(num_doms == target);
    }

}


// ----------------------------------------------------------
TEST(ascent_partition, test_mpi_partition_fields)
{
    Node n;
    ascent::about(n);

    // only run this test if ascent was built with hdf5 +  vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled" ||
       n["runtimes/ascent/hdf5/status"].as_string() == "disabled" )
    {
        ASCENT_INFO("Ascent Rendering and/or HDF5 support are disabled, skipping test");
        return;
    }

    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mpi::mesh::examples::spiral_round_robin(NUM_DOMAINS,data,comm);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing blueprint partition of multi-domain mesh with MPI");
    
    string output_path;
    if(par_rank == 0)
    {
        output_path = prepare_output_dir();
    }
    else
    {
        output_path = output_dir();
    }

    string output_base = conduit::utils::join_file_path(output_path,
                                                        "tout_partition_target_1_fields");
    string output_root = output_base + "_result.cycle_000000.root";

    if(par_rank == 0)
    {
        // remove existing file
        if(utils::is_file(output_root))
        {
            utils::remove_file(output_root);
        }
    }

    conduit::Node actions;
    int target = 1;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"]  = "partition";
    pipelines["pl1/f1/params/target"] = target;
    pipelines["pl1/f1/params/fields"].append()="dist";

    //add the extract
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_base + "_result";
    extracts["e1/params/protocol"] = "hdf5";

    extracts["einput/type"] = "relay";
    extracts["einput/params/path"] = output_base + "_input";
    extracts["einput/params/protocol"] = "hdf5";

    // Add a scene that shows domain id
    //
    //add the scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "dist";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = conduit::utils::join_file_path(output_path, "tout_partition_target_1_fields_result_render");

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    MPI_Barrier(comm);

    if(par_rank == 0)
    {
      EXPECT_TRUE(conduit::utils::is_file(output_root));
      Node read_mesh;
      conduit::relay::io::blueprint::load_mesh(output_root,read_mesh);

      int num_doms = conduit::blueprint::mesh::number_of_domains(read_mesh);
      EXPECT_TRUE(num_doms == target);
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


