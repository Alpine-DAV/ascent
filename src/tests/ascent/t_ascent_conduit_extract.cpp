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
#include <ascent_string_utils.hpp>

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
TEST(ascent_conduit_extract, test_extract_path)
{
    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              25,
                                              25,
                                              25,
                                              mesh);

    string output_path = prepare_output_dir();
    string image_prefix = "output_path_{family:05d}_{cycle:04d}_{time:0.4f}";
    const string output_file = conduit::utils::join_file_path(output_path,image_prefix);

    string image_prefix_only_format = "output_path_%03d_only_format";
    const string output_file_only_format = conduit::utils::join_file_path(output_path,image_prefix_only_format);

    string image_prefix_no_format = "output_path_no_format_";
    const string output_file_no_format = conduit::utils::join_file_path(output_path,image_prefix_no_format);
    remove_test_image(output_file);

    string extract_prefix = "output_path_{family:05d}_{cycle:04d}_{time:0.4f}";
    const string extract_file = conduit::utils::join_file_path(output_path,extract_prefix);

    // Use Ascent to export our mesh to blueprint flavored hdf5 files
    Ascent a;

    // open ascent
    a.open();

    // publish mesh to ascent
    a.publish(mesh);

    // setup actions
    Node actions;
    Node &add_act2 = actions.append();
    add_act2["action"] = "add_scenes";

    Node &scenes = add_act2["scenes"];
    // Showing family value incrementation:
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;
    scenes["s2/plots/p1/type"] = "pseudocolor";
    scenes["s2/plots/p1/field"] = "braid";
    scenes["s2/image_prefix"] = output_file;

    // Showing formatting with only a format field:
    scenes["s3/plots/p1/type"] = "pseudocolor";
    scenes["s3/plots/p1/field"] = "braid";
    scenes["s3/image_prefix"] = output_file_only_format;

    // Showing that family value is added to output file names when no other format given
    scenes["s4/plots/p1/type"] = "pseudocolor";
    scenes["s4/plots/p1/field"] = "braid";
    scenes["s4/image_prefix"] = output_file_no_format;

    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/params/path"] = extract_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
    extracts["e1/params/fields"].append().set("braid");

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // execute the actions
    a.execute(actions);

    // close ascent
    a.close();


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


