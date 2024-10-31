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

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    oss << "tout_partition_target_1_mpi";
    string output_base = conduit::utils::join_file_path(output_path,
                                                        oss.str());
    std::ostringstream ossjson;
    ossjson << "tout_partition_target_1_mpi_json";
    string output_json = conduit::utils::join_file_path(output_base,
		    					ossjson.str());
    // remove existing file
    if(utils::is_file(output_base))
    {
        utils::remove_file(output_base);
    }
    if(utils::is_file(output_json))
    {
        utils::remove_file(output_json);
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
    extracts["e1/params/path"] = output_base;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    //Two files in _output directory:
    //tout_partition_multi_dom_serial
    //tout_partition_multi_dom_serial_json
    if(par_rank == 0)
    {
      EXPECT_TRUE(conduit::utils::is_file(output_base));
      Node read_csv;
      conduit::relay::io::load(output_base,read_csv);

      int num_doms = conduit::blueprint::mesh::number_of_domains(read_csv);
      EXPECT_TRUE(num_doms == target);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_partition, test_mpi_partition_target_10)
{
    Node n;
    ascent::about(n);

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

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    oss << "tout_partition_target_10_mpi";
    string output_base = conduit::utils::join_file_path(output_path,
                                                        oss.str());
    std::ostringstream ossjson;
    ossjson << "tout_partition_taret_10_mpi_json";
    string output_json = conduit::utils::join_file_path(output_base,
		    					ossjson.str());
    // remove existing file
    if(utils::is_file(output_base))
    {
        utils::remove_file(output_base);
    }
    if(utils::is_file(output_json))
    {
        utils::remove_file(output_json);
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
    extracts["e1/params/path"] = output_base;

    //add the scene
    //conduit::Node &add_scenes= actions.append();
    //add_scenes["action"] = "add_scenes";
    //conduit::Node &scenes = add_scenes["scenes"];
    //scenes["s1/plots/p1/type"] = "pseudocolor";
    //scenes["s1/plots/p1/field"] = "dist";
    //scenes["s1/plots/p1/field"] = "rank";
    //scenes["s1/plots/p1/pipeline"] = "pl1";
    //  scenes["s1/plots/p1/color_table/discrete"] = "true";
    //scenes["s1/image_prefix"] = "tout_mpi_partition"; 

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    //Two files in _output directory:
    //tout_partition_multi_dom_serial
    //tout_partition_multi_dom_serial_json
    if(par_rank == 0)
    {
      EXPECT_TRUE(conduit::utils::is_file(output_base));
      Node read_csv;
      conduit::relay::io::load(output_base,read_csv);

      int num_doms = conduit::blueprint::mesh::number_of_domains(read_csv);
      EXPECT_TRUE(num_doms == target);
    }
}


//-----------------------------------------------------------------------------
TEST(ascent_partition, test_mpi_partition_selection)
{
    Node n;
    ascent::about(n);

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

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    oss << "tout_partition_selection_mpi";
    string output_base = conduit::utils::join_file_path(output_path,
                                                        oss.str());
    std::ostringstream ossjson;
    ossjson << "tout_partition_selection_mpi_json";
    string output_json = conduit::utils::join_file_path(output_base,
		    					ossjson.str());
    // remove existing file
    if(utils::is_file(output_base))
    {
        utils::remove_file(output_base);
    }
    if(utils::is_file(output_json))
    {
        utils::remove_file(output_json);
    }

    conduit::Node actions;
    int target = 1;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"]  = "partition";
    pipelines["pl1/f1/params/selections/type"] = "logical";
    //float start[2] = {0,0};
    float start[3] = {0,0,0};
    float end[3] = {10,10,0};
    //float end[2] = {0.5,0.5};
    pipelines["pl1/f1/params/selections/start"].set(start,3); 
    pipelines["pl1/f1/params/selections/end"].set(end,3); 
    
    //add the extract
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"] = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_base;

    //add the scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "dist";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_base; 

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    //Two files in _output directory:
    //tout_partition_multi_dom_serial
    //tout_partition_multi_dom_serial_json
    if(par_rank == 0)
    {
      EXPECT_TRUE(conduit::utils::is_file(output_base));
      Node read_csv;
      conduit::relay::io::load(output_base,read_csv);

      int num_doms = conduit::blueprint::mesh::number_of_domains(read_csv);
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


