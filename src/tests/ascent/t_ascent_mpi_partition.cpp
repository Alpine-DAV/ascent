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

#include "t_config.hpp"
#include "t_utils.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
TEST(ascent_partition, test_partition_2D_multi_dom)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    //
    //Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing blueprint partition of multi-domain mesh with MPI");

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    oss << "tout_partition_multi_dom_mpi";
    string output_base = conduit::utils::join_file_path(output_path,
                                                        oss.str());
    oss << ".csv";
    string output_dir = conduit::utils::join_file_path(output_path,
                                                        oss.str());
    std::ostringstream voss,eoss;
    voss << "vertex_data.csv";
    string output_vertex = conduit::utils::join_file_path(output_dir,
		    					   voss.str());
    eoss << "element_data.csv";
    string output_element = conduit::utils::join_file_path(output_dir,
		    					    eoss.str());
    // remove existing directory
    if(utils::is_directory(output_dir))
    {
        utils::remove_directory(output_dir);
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
    extracts["e1/type"] = "flatten";
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

    int root = 0;
    if(par_rank == root)
    {
        //A directory called tout_partition_multi_dom_serial.csv 
        EXPECT_TRUE(conduit::utils::is_directory(output_dir));
        //Two files in above directory:
        //vertex_data.csv
        //element_data.csv
        EXPECT_TRUE(conduit::utils::is_file(output_vertex));
        EXPECT_TRUE(conduit::utils::is_file(output_element));
        Node read_csv;
        conduit::relay::io::load(output_vertex,read_csv);

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


