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

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing blueprint partition of multi-domain mesh in serial");

    string output_path = prepare_output_dir();
    string output_base = conduit::utils::join_file_path(output_path,
                                                        "tout_partition_multi_dom_serial");
    string output_root = output_base + ".cycle_000000.root";

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
    // pipelines["pl1/f1/type"] = "add_domain_ids";
    // pipelines["pl1/f1/params/output"] = "d_id_pre";

    pipelines["pl1/f2/type"]  = "partition";
    pipelines["pl1/f2/params/target"] = target;

    // pipelines["pl1/f3/type"] = "add_domain_ids";
    // pipelines["pl1/f3/params/output"] = "d_id_post";

    //add the extract
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["eout/type"] = "relay";
    extracts["eout/pipeline"] = "pl1";
    extracts["eout/params/path"] = output_base;
    extracts["eout/params/protocol"] = "hdf5";

    extracts["einput/type"] = "relay";
    extracts["einput/params/path"] = output_base + "_input";
    extracts["einput/params/protocol"] = "hdf5";

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    EXPECT_TRUE(conduit::utils::is_file(output_root));
    Node read_mesh;
    conduit::relay::io::blueprint::load_mesh(output_root,read_mesh);

    int num_doms = conduit::blueprint::mesh::number_of_domains(read_mesh);
    EXPECT_TRUE(num_doms == target);
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    result = RUN_ALL_TESTS();
    return result;
}


