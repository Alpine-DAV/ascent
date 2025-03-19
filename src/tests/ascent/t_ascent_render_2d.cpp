//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_2d.cpp
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

index_t EXAMPLE_MESH_SIDE_DIM = 50;

//-----------------------------------------------------------------------------
// this test checks that we get an exception when ascent was compiled
// w/o any rendering support
//-----------------------------------------------------------------------------
TEST(ascent_render_2d, test_render_2d_default_runtime_non_rendering_support)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built *NOT* with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "enabled")
    {
        ASCENT_INFO("Ascent vtkm support enabled, skipping this test");
        return;
    }

    //
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    //
    // Create the actions.
    //
    Node actions;

    conduit::Node scenes;
    scenes["scene1/plots/plt1/type"]         = "pseudocolor";
    scenes["scene1/plots/plt1/field"] = "braid";
    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    EXPECT_THROW(ascent.execute(actions),conduit::Error);
    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_render_2d, test_render_2d_default_runtime)
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
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                             "tout_render_2d_default_runtime");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node scenes;
    scenes["scene1/plots/plt1/type"]         = "pseudocolor";
    scenes["scene1/plots/plt1/field"] = "braid";
    scenes["scene1/image_prefix"] = output_file;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example rendering a 2d field.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_render_2d, test_render_2d_uniform_default_runtime)
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
    // Create example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                               10,
                                               10,
                                               1,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                    "tout_render_2d_uniform_default_runtime");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node scenes;
    scenes["scene1/plots/plt1/type"]         = "pseudocolor";
    scenes["scene1/plots/plt1/field"] = "braid";
    scenes["scene1/image_prefix"] = output_file;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_render_2d, test_render_2d_render_serial_backend)
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

    ASCENT_INFO("Testing 2D Ascent Runtime");

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_2d_ascent_serial_backend");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node scenes;
    scenes["scene1/plots/plt1/type"]         = "pseudocolor";
    scenes["scene1/plots/plt1/field"] = "braid";
    scenes["scene1/image_prefix"] =  output_file;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["runtime/backend"] = "serial";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

// TODO: Address 2D Zoom issue a follow on PR
//
// //-----------------------------------------------------------------------------
// TEST(ascent_render_2d, test_render_2d_render_serial_backend_zoom)
// {
//
//     // the vtkm runtime is currently our only rendering runtime
//     Node n;
//     ascent::about(n);
//     // only run this test if ascent was built with vtkm support
//     if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
//     {
//         ASCENT_INFO("Ascent vtkm support disabled, skipping test");
//         return;
//     }
//
//     ASCENT_INFO("Testing 2D Ascent Runtime");
//
//     //
//     // Create an example mesh.
//     //
//     Node data, verify_info;
//     conduit::blueprint::mesh::examples::braid("quads",
//                                                EXAMPLE_MESH_SIDE_DIM,
//                                                EXAMPLE_MESH_SIDE_DIM,
//                                                0,
//                                                data);
//
//     EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
//
//     string output_path = prepare_output_dir();
//     string output_file = conduit::utils::join_file_path(output_path, "tout_render_2d_ascent_serial_backend_zoom");
//     // remove old images before rendering
//     remove_test_image(output_file);
//
//     //
//     // Create the actions.
//     //
//     Node actions;
//
//     conduit::Node scenes;
//     scenes["scene1/plots/plt1/type"]         = "pseudocolor";
//     scenes["scene1/plots/plt1/field"] = "braid";
//     scenes["scene1/renders/r1/image_prefix"] =  output_file;
//     scenes["scene1/renders/r1/camera/zoom"] =  .5;
//
//     conduit::Node &add_scenes = actions.append();
//     add_scenes["action"] = "add_scenes";
//     add_scenes["scenes"] = scenes;
//
//     //
//     // Run Ascent
//     //
//
//     Ascent ascent;
//     Node ascent_opts;
//     // default is now ascent
//     ascent_opts["runtime/type"] = "ascent";
//     ascent_opts["runtime/backend"] = "serial";
//     ascent.open(ascent_opts);
//     ascent.publish(data);
//     ascent.execute(actions);
//     ascent.close();
//
//     // check that we created an image
//     EXPECT_TRUE(check_test_image(output_file));
// }


//-----------------------------------------------------------------------------
TEST(ascent_render_2d, test_render_2d_uniform_render_serial_backend)
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

    ASCENT_INFO("Testing 2D Ascent Runtime");

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_2d_uniform_ascent_serial_backend");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node scenes;
    scenes["scene1/plots/plt1/type"]         = "pseudocolor";
    scenes["scene1/plots/plt1/field"] = "braid";
    scenes["scene1/image_prefix"] =  output_file;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["runtime/backend"] = "serial";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_render_2d, test_render_2d_cam)
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

    ASCENT_INFO("Testing 2D Ascent Runtime 2d camera controls");

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                               EXAMPLE_MESH_SIDE_DIM,
                                               EXAMPLE_MESH_SIDE_DIM,
                                               0,
                                               data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file_base = conduit::utils::join_file_path(output_path, "tout_render_2d_uniform_2d_cam");
    string output_file_v1 = output_file_base  + "_view_1_";
    string output_file_v2 = output_file_base  + "_view_2_";
    string output_file_v3 = output_file_base  + "_view_3_";
    string output_file_v4 = output_file_base  + "_view_4_";
    string output_info = output_file_base  + "_info";
    // remove old images before rendering
    remove_test_image(output_file_v1);
    remove_test_image(output_file_v2);
    remove_test_image(output_file_v3);
    remove_test_image(output_file_v4);

    //
    // Create the actions.
    //
    Node actions;
    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    conduit::Node &scenes = add_scenes["scenes"];
    scenes["scene1/plots/plt1/type"] = "pseudocolor";
    scenes["scene1/plots/plt1/field"] = "braid";

    scenes["scene1/renders/r1/image_prefix"] =  output_file_v1;
    scenes["scene1/renders/r1/camera/2d"] = {-10.0,10.0,-10.0,10.0};

    scenes["scene1/renders/r2/image_prefix"] =  output_file_v2;
    scenes["scene1/renders/r2/camera/2d"] = {-20.0,20.0,-20.0,20.0};

    scenes["scene1/renders/r3/image_prefix"] =  output_file_v3;
    scenes["scene1/renders/r3/camera/2d"] = {-7.0,3.0,0.0,4.0};

    scenes["scene1/renders/r4/image_prefix"] = output_file_v4;
    scenes["scene1/renders/r4/camera/2d"] = {-10.0,0.0,-10.0,10.0};
    scenes["scene1/renders/r4/image_width"]  = 512;
    scenes["scene1/renders/r4/image_height"] = 1024;


    conduit::Node info;

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(info);
    ascent.close();

    std::cout << info.to_yaml() << std::endl;

    // check output
    EXPECT_TRUE(check_test_image(output_file_v1));
    EXPECT_TRUE(check_test_image(output_file_v2));
    EXPECT_TRUE(check_test_image(output_file_v3));
    EXPECT_TRUE(check_test_image(output_file_v4));
}


//-----------------------------------------------------------------------------
TEST(ascent_render_2d, test_render_2d_bentgrid_example)
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

    ASCENT_INFO("Testing 2D Ascent Runtime");

    //
    // load example mesh
    //
    Node data,verify_info;
    conduit::relay::io::blueprint::read_mesh(test_data_file("bentgrid_2d_visitghost_yaml.root"),
                                             data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_2d_bentgrid");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    Node actions;

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "add_domain_ids";
    // filter knobs all these are optional
    conduit::Node &domainId_params = pipelines["pl1/f1/params"];
    domainId_params["output"] = "domain_ids";   // largest value on the x-axis

    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    conduit::Node scenes;
    scenes["scene1/plots/plt1/type"] = "pseudocolor";
    scenes["scene1/plots/plt1/field"] = "domain_ids";
    scenes["scene1/plots/plt1/pipeline"] = "pl1";
    scenes["scene1/renders/r1/image_prefix"] =  output_file;
    scenes["scene1/renders/r1/camera/elevation"] = 10.;
    scenes["scene1/renders/r1/camera/azimuth"] =  -45.;

    conduit::Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    // default is now ascent
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
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


