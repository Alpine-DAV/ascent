//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_relay.cpp
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


index_t EXAMPLE_MESH_SIDE_DIM = 10;

//-----------------------------------------------------------------------------
TEST(ascent_conduit_extract, test_pass_thru)
{
    Node n;
    ascent::about(n);

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    data["state/domain_id"] = 0;

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing conduit  extract in serial");
    
    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    // add the extract
    extracts["e1/type"]  = "conduit";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    conduit::Node & info =  ascent.info();

    // copy out our extract
    conduit::Node extract_copy;
    extract_copy.set(info["extracts"][0]);

    ascent.close();
    // diff to make sure data looks as we expect
    Node diff_info;
    EXPECT_FALSE(extract_copy["data"][0].diff(data,diff_info));
}

//-----------------------------------------------------------------------------
TEST(ascent_conduit_extract, test_pipeline_result)
{
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
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing slice to in-memory extract");

    //
    // Create the actions.
    //
    // slice + conduit in memory extract
    conduit::Node actions;
    // add the pipeline

    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];

    // pipeline 1
    pipelines["pl1/f1/type"] = "slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x"] = 0.f;
    slice_params["point/y"] = 0.f;
    slice_params["point/z"] = 0.f;

    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 1.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    // add the extract
    extracts["e1/type"]  = "conduit";
    extracts["e1/pipeline"] = "pl1";

    std::cout << actions.to_yaml() << std::endl;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    conduit::Node & info = ascent.info();

    // copy out our extract
    conduit::Node extract_copy;
    extract_copy.set(info["extracts"][0]);

    ascent.close();

    // pass back copy and render the result

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                            "tout_in_memory_extract_render_slice_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    actions.reset();

    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes  = add_scenes["scenes"];

    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/image_prefix"] = output_file;

    ascent.open();
    ascent.publish(extract_copy["data"]);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));

}



//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    if(argc == 2)
    {
        EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
    }

    result = RUN_ALL_TESTS();
    return result;
}


