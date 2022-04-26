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

    int root = 0;
    if(par_rank == root)
    	ASCENT_INFO("Testing blueprint partition of multi-domain mesh with MPI");

    string output_path = prepare_output_dir();
    std::ostringstream oss;

    oss << "tout_partition_multi_dom_mpi";
    string output_base = conduit::utils::join_file_path(output_path,
                                                        oss.str());
    std::ostringstream ossjson;
    ossjson << "tout_partition_multi_dom_mpi_json";
    string output_json = conduit::utils::join_file_path(output_base,
		    					ossjson.str());
    // remove existing directory
    if(par_rank == root)
    {
    	if(utils::is_file(output_base))
    	{
    	    utils::remove_file(output_base);
    	}
    	if(utils::is_file(output_json))
    	{
    	    utils::remove_file(output_json);
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

    if(par_rank == root)
    {
        //Two files in _output directory:
        //tout_partition_multi_dom_mpi
        //tout_partition_multi_dom_mpi_json
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


