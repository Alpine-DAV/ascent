//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_slice.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;


//-----------------------------------------------------------------------------
TEST(ascent_slice, test_slice)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing slice");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_slice_3d");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x"] = 0.f;
    slice_params["point/y"] = 0.f;
    slice_params["point/z"] = 0.f;

    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 0.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of the slice filter with a single plane.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_slice, test_exaslice)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing slice");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_exaslice_3d");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "exaslice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x"] = 0.f;
    slice_params["point/y"] = 0.f;
    slice_params["point/z"] = 0.f;

    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 0.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of the slice filter with a single plane.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_slice, test_slice_offset)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing slice");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_slice_offset_3d");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x_offset"] = 0.f;
    slice_params["point/y_offset"] = -0.5f;
    slice_params["point/z_offset"] = 0.f;

    slice_params["normal/x"] = 1.f;
    slice_params["normal/y"] = 1.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of the slice filter with a single plane.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_slice, test_slice_off_axis)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing slice off axis");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_slice_3d_off_axis");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x"] = 1.f;
    slice_params["point/y"] = 1.f;
    slice_params["point/z"] = 1.f;

    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 0.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file, 0.01f));
    std::string msg = "An example of the slice filter with a single plane (off-axis).";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
TEST(ascent_slice, test_3slice)
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

    ASCENT_INFO("Testing 3slice");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_3slice_3d");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "3slice";
    // filter knobs (all these are optional)

    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["x_offset"] = 1.f;   // largest value on the x-axis
    slice_params["y_offset"] = 0.f;   // middle of the y-axis
    slice_params["z_offset"] = -1.f;  // smalles value of the z-axis

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";

    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of the three slice filter.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_slice, test_auto_slice_z_axis)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing automatic slice with z-axis and 10 levels");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_auto_slice_z_axis");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "auto_slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["field"]    = "braid";
    slice_params["levels"]   = 10;
    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 0.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]     = "pseudocolor";
    scenes["s1/plots/p1/field"]    = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"]      = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file, 0.01f));
    std::string msg = "An example of the automaic slice filter using a z-axis normal, 10 levels, and the default camera.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_slice, test_auto_slice_x_axis)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing automatic slice with x-axis and 10 levels");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_auto_slice_x_axis");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "auto_slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["field"]    = "braid";
    slice_params["levels"]   = 10;
    slice_params["normal/x"] = 1.f;
    slice_params["normal/y"] = 0.f;
    slice_params["normal/z"] = 0.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]             = "pseudocolor";
    scenes["s1/plots/p1/field"]            = "braid";
    scenes["s1/plots/p1/pipeline"]         = "pl1";
    scenes["s1/renders/r1/camera/azimuth"] = 90.0;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file, 0.01f));
    std::string msg = "An example of the automaic slice filter using an x-axis normal, 10 levels, and an adusted camera.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_slice, test_auto_slice_y_axis)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing automatic slice with y-axis and 10 levels");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_auto_slice_y_axis");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "auto_slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["field"]    = "braid";
    slice_params["levels"]   = 10;
    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 1.f;
    slice_params["normal/z"] = 0.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]               = "pseudocolor";
    scenes["s1/plots/p1/field"]              = "braid";
    scenes["s1/plots/p1/pipeline"]           = "pl1";
    scenes["s1/renders/r1/camera/elevation"] = 90.0;
    scenes["s1/renders/r1/image_prefix"]     = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file, 0.01f));
    std::string msg = "An example of the automaic slice filter using a y-axis normal, 10 levels, and an adusted camera.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_slice, test_auto_slice_xy_axis)
{
    // the vtkm runtime is currently our only rendering runtime
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

    ASCENT_INFO("Testing automatic slice with an xy-axis and 10 levels");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_auto_slice_xy_axis");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "auto_slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["field"]    = "braid";
    slice_params["levels"]   = 10;
    slice_params["normal/x"] = 1.f;
    slice_params["normal/y"] = 1.f;
    slice_params["normal/z"] = 0.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]               = "pseudocolor";
    scenes["s1/plots/p1/field"]              = "braid";
    scenes["s1/plots/p1/pipeline"]           = "pl1";
    scenes["s1/renders/r1/camera/azimuth"]   = 90.0;
    scenes["s1/renders/r1/camera/elevation"] = 45.0;
    scenes["s1/renders/r1/image_prefix"]     = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file, 0.01f));
    std::string msg = "An example of the automaic slice filter using an xy-axis normal, 10 levels, and an adusted camera.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


// implicit func slice cases

//-----------------------------------------------------------------------------
TEST(ascent_slice, test_sphere_slice)
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
    ASCENT_INFO("Testing implicit sphere slice");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_slice_sphere_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node & pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "slice";
    conduit::Node &slice_params = pipelines["pl1/f1/params"];

    slice_params["sphere/center/x"] = 0.0;
    slice_params["sphere/center/y"] = 0.0;
    slice_params["sphere/center/z"] = 0.0;
    slice_params["sphere/radius"] = 10.0;

    // add a scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    // add an extract
    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = output_file + "_hdf5";
    extracts["e1/params/protocol"] = "hdf5";

    // run ascent
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of a spherical slice.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_slice, test_cyln_slice)
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
    ASCENT_INFO("Testing implicit cylinder slice");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_slice_cylinder_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node & pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "slice";
    conduit::Node &slice_params = pipelines["pl1/f1/params"];

    slice_params["cylinder/center/x"] = 0.0;
    slice_params["cylinder/center/y"] = 0.0;
    slice_params["cylinder/center/z"] = 0.0;
    slice_params["cylinder/axis/x"] = 0.0;
    slice_params["cylinder/axis/y"] = 0.0;
    slice_params["cylinder/axis/z"] = 1.0;
    slice_params["cylinder/radius"] = 10.0;

    // add a scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    // add an extract
    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = output_file + "_hdf5";
    extracts["e1/params/protocol"] = "hdf5";

    // run ascent
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of a spherical slice.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_slice, test_box_slice)
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
    ASCENT_INFO("Testing implicit box slice");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_slice_box_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node & pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "slice";
    conduit::Node &slice_params = pipelines["pl1/f1/params"];

    slice_params["box/min/x"] = 0.0;
    slice_params["box/min/y"] = 0.0;
    slice_params["box/min/z"] = 0.0;
    slice_params["box/max/x"] = 20.0;
    slice_params["box/max/y"] = 15.0;
    slice_params["box/max/z"] = 5.0;
    
    // add a scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    // add an extract
    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = output_file + "_hdf5";
    extracts["e1/params/protocol"] = "hdf5";

    // run ascent
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of a spherical slice.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_slice, test_plane_slice)
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
    ASCENT_INFO("Testing implicit plane slice");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_slice_plane_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node & pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "slice";
    conduit::Node &slice_params = pipelines["pl1/f1/params"];

    slice_params["plane/point/x"] = 0.0;
    slice_params["plane/point/y"] = 0.0;
    slice_params["plane/point/z"] = 0.0;
    slice_params["plane/normal/x"] = 1.0;
    slice_params["plane/normal/y"] = 0.0;
    slice_params["plane/normal/z"] = 1.0;

    // add a scene
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    // add an extract
    conduit::Node &add_extracts= actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"]  = "pl1";
    extracts["e1/params/path"] = output_file + "_hdf5";
    extracts["e1/params/protocol"] = "hdf5";

    // run ascent
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of a spherical slice.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


// TODO: We want this case to work, but we may have a VTK-m issue
// //-----------------------------------------------------------------------------
// TEST(ascent_slice, test_slice_plane_of_plane)
// {
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with vtkm support
//     if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent vtkm support disabled, skipping test");
//         return;
//     }
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     // simple plane
//     data["coordsets/coords/type"] = "explicit";
//     Node &coords = data["coordsets/coords/values"];
//
//     coords["x"] = {-2.0, -1.0, 0.0, 1.0, 2.0,
//                    -2.0, -1.0, 0.0, 1.0, 2.0,
//                    -2.0, -1.0, 0.0, 1.0, 2.0};
//
//     coords["y"] = {-2.0, -2.0, -2.0, -2.0, -2.0,
//                     0.0, 0.0, 0.0, 0.0, 0.0,
//                     2.0,  2.0, 2.0, 2.0, 2.0};
//
//     coords["z"] = {0.0, 0.0, 0.0, 0.0, 0.0,
//                    0.0, 0.0, 0.0, 0.0, 0.0,
//                    0.0, 0.0, 0.0, 0.0, 0.0};
//
//     data["topologies/topo/type"] = "unstructured";
//     data["topologies/topo/coordset"] = "coords";
//     data["topologies/topo/elements/shape"] = "quad";
//     data["topologies/topo/elements/connectivity"] =  {0,5,6,1,
//                                                       1,6,7,2,
//                                                       2,7,8,3,
//                                                       3,8,9,4,
//                                                       5,10,11,6,
//                                                       6,11,12,7,
//                                                       7,12,13,8,
//                                                       8,13,14,9};
//
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//     ASCENT_INFO("Testing implicit plane slice of plane");
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_slice_plane_of_plane_3d");
//
//     // remove old images before rendering
//     remove_test_image(output_file);
//
//     //
//     // Create the actions.
//     //
//     conduit::Node actions;
//     conduit::Node &add_pipelines = actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     conduit::Node & pipelines = add_pipelines["pipelines"];
//
//     pipelines["pl1/f1/type"] = "slice";
//     conduit::Node &slice_params = pipelines["pl1/f1/params"];
//
//     slice_params["plane/point/x"] = 0.0;
//     slice_params["plane/point/y"] = 0.0;
//     slice_params["plane/point/z"] = 0.0;
//     slice_params["plane/normal/x"] = 0.0;
//     slice_params["plane/normal/y"] = 1.0;
//     slice_params["plane/normal/z"] = 1.0;
//
//     // add a scene
//     conduit::Node &add_scenes= actions.append();
//     add_scenes["action"] = "add_scenes";
//     conduit::Node &scenes = add_scenes["scenes"];
//     scenes["s1/plots/p1/type"]  = "pseudocolor";
//     scenes["s1/plots/p1/field"] = "radial";
//     scenes["s1/plots/p1/pipeline"] = "pl1";
//     scenes["s1/image_prefix"] = output_file;
//
//     // add an extract
//     conduit::Node &add_extracts= actions.append();
//     add_extracts["action"] = "add_extracts";
//     conduit::Node &extracts = add_extracts["extracts"];
//     extracts["e1/type"]  = "relay";
//     extracts["e1/params/path"] = output_file + "_hdf5_input";
//     extracts["e1/params/protocol"] = "hdf5";
//
//     extracts["e2/type"]  = "relay";
//     extracts["e2/pipeline"]  = "pl1";
//     extracts["e2/params/path"] = output_file + "_hdf5_res";
//     extracts["e2/params/protocol"] = "hdf5";
//
//     // run ascent
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     EXPECT_TRUE(check_test_image(output_file));
//     std::string msg = "An example of a spherical slice.";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }



// TODO: We want this case to work, but we may have a VTK-m issue
// //-----------------------------------------------------------------------------
// TEST(ascent_slice, test_cyl_plane_slice)
// {
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with vtkm support
//     if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent vtkm support disabled, skipping test");
//         return;
//     }
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("hexs",
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               EXAMPLE_MESH_SIDE_DIM,
//                                               data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//     ASCENT_INFO("Testing implicit plane slice");
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path,"tout_slice_cyl_then_plane_3d");
//
//     // remove old images before rendering
//     remove_test_image(output_file);
//
//     //
//     // Create the actions.
//     //
//     conduit::Node actions;
//     conduit::Node &add_pipelines = actions.append();
//     add_pipelines["action"] = "add_pipelines";
//     conduit::Node & pipelines = add_pipelines["pipelines"];
//
//     pipelines["pl1/f1/type"] = "slice";
//     conduit::Node &slice_params1 = pipelines["pl1/f1/params"];
//
//     slice_params1["cylinder/center/x"] = 0.0;
//     slice_params1["cylinder/center/y"] = 0.0;
//     slice_params1["cylinder/center/z"] = 0.0;
//     slice_params1["cylinder/axis/x"] = 0.0;
//     slice_params1["cylinder/axis/y"] = 0.0;
//     slice_params1["cylinder/axis/z"] = 1.0;
//     slice_params1["cylinder/radius"] = 10.0;
//
//     pipelines["pl1/f2/type"] = "slice";
//     conduit::Node &slice_params2 = pipelines["pl1/f2/params"];
//
//     slice_params2["plane/point/x"] = 0.0;
//     slice_params2["plane/point/y"] = 0.0;
//     slice_params2["plane/point/z"] = 0.0;
//     slice_params2["plane/normal/x"] = 0.0;
//     slice_params2["plane/normal/y"] = 0.0;
//     slice_params2["plane/normal/z"] = 1.0;
//
//     // add a scene
//     conduit::Node &add_scenes= actions.append();
//     add_scenes["action"] = "add_scenes";
//     conduit::Node &scenes = add_scenes["scenes"];
//     scenes["s1/plots/p1/type"]  = "pseudocolor";
//     scenes["s1/plots/p1/field"] = "radial";
//     scenes["s1/plots/p1/pipeline"] = "pl1";
//     scenes["s1/image_prefix"] = output_file;
//
//     // add an extract
//     conduit::Node &add_extracts= actions.append();
//     add_extracts["action"] = "add_extracts";
//     conduit::Node &extracts = add_extracts["extracts"];
//     extracts["e1/type"]  = "relay";
//     extracts["e1/pipeline"]  = "pl1";
//     extracts["e1/params/path"] = output_file + "_hdf5";
//     extracts["e1/params/protocol"] = "hdf5";
//
//     // run ascent
//     Ascent ascent;
//     ascent.open();
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     EXPECT_TRUE(check_test_image(output_file));
//     std::string msg = "An example of a spherical slice.";
//     ASCENT_ACTIONS_DUMP(actions,output_file,msg);
// }
//
//
//



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


