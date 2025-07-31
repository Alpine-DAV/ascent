//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_3d.cpp
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


index_t EXAMPLE_MESH_SIDE_DIM = 20;
//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_render_default_runtime)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_default_runtime");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

TEST(ascent_render_3d, test_render_3d_original_bounds)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file =
      conduit::utils::join_file_path(output_path,"tout_render_3d_original_bounds");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "clip";
    // filter knobs
    conduit::Node &clip_params = pipelines["pl1/f1/params"];
    clip_params["box/min/x"] = -10.1;
    clip_params["box/min/y"] = -10.1;
    clip_params["box/min/z"] = 0.;
    clip_params["box/max/x"] = 10.01; // <=
    clip_params["box/max/y"] = 10.01;
    clip_params["box/max/z"] = 10.01;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/renders/r1/image_prefix"] = output_file;
    scenes["s1/renders/r1/use_original_bounds"] = "true";
    scenes["s1/renders/r1/camera/azimuth"] = 90;


    conduit::Node exs;
    exs["e1/type"]         = "vtk";
    exs["e1/params/path"]         = output_file + "_before_vtk";

    exs["e2/type"]         = "vtk";
    exs["e2/pipeline"]     = "pl1";
    exs["e2/params/path"]         = output_file + "_after_vtk";

    conduit::Node actions;

    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    conduit::Node &add_plots = actions.append();
    add_plots["scenes"] = scenes;
    add_plots["action"] = "add_scenes";

    conduit::Node &add_exs = actions.append();
    add_exs["extracts"] = exs;
    add_exs["action"] = "add_extracts";


    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

TEST(ascent_render_3d, test_render_3d_single_comp_scalar)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    data["fields/braid2"] = data["fields/braid"];
    data["fields/braid2/values/c0"] = data["fields/braid/values"];
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_single_comp");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid2";
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

TEST(ascent_render_3d, test_render_3d_points)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }


    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("points",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with points");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_points");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file,0.09f));
}

TEST(ascent_render_3d, test_render_3d_points_const_radius)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }


    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("points",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with points");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_points_const_radius");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/points/radius"] = 1.f;
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file,0.01f));
    std::string msg = "An example of rendering a point field with constant radius.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

TEST(ascent_render_3d, test_render_3d_points_variable_radius)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }


    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("points",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with points");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_points_variable_radius");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/points/radius"] = 0.25f;
    // this detla is relative to the base radius
    scenes["s1/plots/p1/points/radius_delta"] = 2.0f;
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM default
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file, 0.09));
    std::string msg = "An example of rendering a point field with variable radius.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

TEST(ascent_render_3d, test_render_3d_bg_fg_color)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with custom bg/fg colors");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_bg_fg_colors");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/renders/r1/image_prefix"]   = output_file;
    float bg_color[3] = {1.f, 1.f, 1.f};
    float fg_color[3] = {0.f, 0.f, 0.f};
    scenes["s1/renders/r1/bg_color"].set(bg_color,3);
    scenes["s1/renders/r1/fg_color"].set(fg_color,3);

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of rendering custom background and foreground colors.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

TEST(ascent_render_3d, test_render_3d_no_annotations)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with no_annotations");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_no_annotations");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/renders/r1/image_prefix"]  = output_file;
    scenes["s1/renders/r1/annotations"] = "false";

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of rendering with no annotations.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


TEST(ascent_render_3d, test_render_3d_no_world_annotations)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with no world annotations");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_no_world_annotations");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/renders/r1/image_prefix"]  = output_file;
    scenes["s1/renders/r1/world_annotations"] = "false";

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of rendering with no world annotations.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

TEST(ascent_render_3d, test_render_3d_no_screen_annotations)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with no screen annotations");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_no_screen_annotations");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/renders/r1/image_prefix"]  = output_file;
    scenes["s1/renders/r1/screen_annotations"] = "false";

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of rendering with no screen annotations.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}



TEST(ascent_render_3d, test_render_3d_name_format)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with image name format");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_name_format");

    // remove old images before rendering
    remove_test_image(output_file, 100, "%04d");


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/renders/r1/image_prefix"]  = output_file + "_%04d";
    scenes["s1/renders/r1/annotations"] = "false";

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.0001f, 100, "%04d"));
    std::string msg = "An example of rendering to a filename using format specifiers.";
    ASCENT_ACTIONS_DUMP_CYCLE_FMT(actions,output_file,msg,100,"%04d");
}

TEST(ascent_render_3d, test_render_3d_name_format_keywords)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              25,
                                              25,
                                              25,
                                              mesh);

    string output_path = prepare_output_dir();
    string image_prefix = "t_output_path_{family:05d}_{cycle:04d}_{time:0.4f}";
    const string output_file = conduit::utils::join_file_path(output_path,image_prefix);
    const string output_file_final_1 = conduit::utils::join_file_path(output_path,"t_output_path_00200_0100_3.1415.png");
    const string output_file_final_2 = conduit::utils::join_file_path(output_path,"t_output_path_00201_0100_3.1415.png");

    string image_prefix_only_format = "t_output_path_%03d_only_format";
    const string output_file_only_format = conduit::utils::join_file_path(output_path,image_prefix_only_format);
    const string output_file_only_format_final = conduit::utils::join_file_path(output_path,"t_output_path_100_only_format.png");

    string image_prefix_no_format = "t_output_path_no_format_";
    const string output_file_no_format = conduit::utils::join_file_path(output_path,image_prefix_no_format);
    const string output_file_no_format_final = conduit::utils::join_file_path(output_path,"t_output_path_no_format_000100.png");

    remove_test_image(output_file_final_1);
    remove_test_image(output_file_final_2);
    remove_test_image(output_file_only_format_final);
    remove_test_image(output_file_no_format_final);

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

    // print our full actions tree
    std::cout << actions.to_yaml() << std::endl;

    // Use Ascent to export our mesh to blueprint flavored hdf5 files
    Ascent ascent;

    // open ascent
    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["family_value_seed"] = 200;
    ascent.open(ascent_opts);

    // publish mesh to ascent
    ascent.publish(mesh);

    // execute the actions
    ascent.execute(actions);

    // close ascent
    ascent.close();

    EXPECT_TRUE(conduit::utils::is_file(output_file_final_1));
    EXPECT_TRUE(conduit::utils::is_file(output_file_final_2));
    EXPECT_TRUE(conduit::utils::is_file(output_file_only_format_final));
    EXPECT_TRUE(conduit::utils::is_file(output_file_no_format_final));
}

TEST(ascent_render_3d, test_render_output_dir_not_exist)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    ASCENT_INFO("Test to verify that Ascent will catch non-existant output paths.\n");

    Node mesh;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              25,
                                              25,
                                              25,
                                              mesh);

    string output_path = prepare_output_dir();
    string image_prefix = "toutput_path_";
    string image_prefix_dir = conduit::utils::join_file_path("non_existant_dir", image_prefix);
    const string output_file = conduit::utils::join_file_path(output_path,image_prefix_dir);

    remove_test_image(output_file);

    // setup actions
    Node actions;
    Node &add_act2 = actions.append();
    add_act2["action"] = "add_scenes";

    Node &scenes = add_act2["scenes"];
    // Showing family value incrementation:
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    // Use Ascent to export our mesh to blueprint flavored hdf5 files
    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["field_filtering"] = "true";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(mesh);

    bool error_occured = false;
    try
    {
        ascent.execute(actions);
    }
    catch(conduit::Error &err)
    {
        if (err.message().find("Error: The specified directory") != std::string::npos)
        {
            error_occured = true;
        }
        else
        {
            std::cout << "The error that was thrown did not match the expected "
                      << "'Error: The specified directory' error" << std::endl;

            std::cout << err.message() << std::endl;
        }
    }

    ascent.close();

    EXPECT_TRUE(error_occured);
}

TEST(ascent_render_3d, test_render_3d_no_bg)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with no background");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_no_bg");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/renders/r1/image_prefix"]  = output_file;
    scenes["s1/renders/r1/render_bg"] = "false";

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of rendering with no background (alpha channel = 0)";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

TEST(ascent_render_3d, test_render_3d_render_azimuth)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_azimuth");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/renders/r1/camera/azimuth"] = 1.;
    scenes["s1/renders/r1/image_prefix"]   = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of changing the azimuth of the camera.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_multi_render_default_runtime)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_multi_default_runtime");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    contour_params["iso_values"] = 0.;

    conduit::Node scenes;
    // plot 1
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    //plot 2
    scenes["s1/plots/p2/type"]         = "volume";
    scenes["s1/plots/p2/field"] = "braid";
    scenes["s1/plots/p2/min_value"]    = -.5;
    scenes["s1/plots/p2/max_value"]    = .5;
    scenes["s1/plots/p2/color_table/name"]  = "rainbow desaturated";

    conduit::Node control_points;

    conduit::Node &point4 = control_points.append();
    point4["type"] = "alpha";
    point4["position"] = 0.;
    point4["alpha"] = 0.0;

    conduit::Node &point5 = control_points.append();
    point5["type"] = "alpha";
    point5["position"] = 1.0;
    point5["alpha"] = .5;

    scenes["s1/plots/p2/color_table/control_points"]  = control_points;

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
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of creating a transfer function for volume rendering.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_render_mesh)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_mesh");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    //plot 1
    scenes["s1/plots/p1/type"] = "mesh";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    float tolerance = 0.01f;
    EXPECT_TRUE(check_test_image(output_file, tolerance));
    std::string msg = "An example of creating a mesh plot.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_multi_render_mesh)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_multi_mesh");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "contour";
    // filter knobs
    conduit::Node &contour_params = pipelines["pl1/f1/params"];
    contour_params["field"] = "braid";
    contour_params["iso_values"] = 0.;

    conduit::Node scenes;
    // plot 1
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "radial";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    //plot 2
    scenes["s1/plots/p2/type"] = "mesh";
    scenes["s1/plots/p2/pipeline"] = "pl1";

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
    //ascent_opts["ascent_info"] = "verbose";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    float tolerance = 0.01f;
    EXPECT_TRUE(check_test_image(output_file, tolerance));
    std::string msg = "An example of creating a mesh plot of a contour.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_render_ascent_serial_backend_uniform)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D serial test");
        return;
    }

    ASCENT_INFO("Testing 3D Rendering with Ascent runtime using Serial Backend");

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_3d_ascent_serial_backend_uniform");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
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
TEST(ascent_render_3d, test_render_3d_render_ascent_serial_backend)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D serial test");
        return;
    }

    ASCENT_INFO("Testing 3D Rendering with Ascent runtime using Serial Backend");

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

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_3d_ascent_serial_backend");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
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
TEST(ascent_render_3d, test_render_3d_render_ascent_min_max)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D serial test");
        return;
    }

    ASCENT_INFO("Testing 3D Rendering with Ascent runtime");

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


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_3d_ascent_min_max");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/min_value"] = -0.5;
    scenes["s1/plots/p1/max_value"] = 0.5;
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

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
    std::string msg = "An example of creating a plot specifying the min and max "
                      "values of the scalar range.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_render_ascent_openmp_backend)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D Ascent-openmp test");
        return;
    }

    if(n["runtimes/ascent/vtkm/backends/openmp"].as_string() != "enabled")
    {
        ASCENT_INFO("Ascent openmp support disabled, skipping 3D Ascent-opemp test");
        return;
    }

    ASCENT_INFO("Testing 3D Rendering with Ascent using OpenMP Backend");

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


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_3d_ascent_openmp_backend");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["runtime/backend"] = "openmp";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}


//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_3d_render_ascent_runtime_cuda_backend)
{

    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D Ascent-cuda test");
        return;
    }

    if(n["runtimes/ascent/vtkm/backends/cuda"].as_string() != "enabled")
    {
        ASCENT_INFO("Ascent CUDA support disabled, skipping 3D Ascent-cuda test");
        return;
    }

    ASCENT_INFO("Testing 3D Rendering with Ascent runtime  using CUDA Backend");

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



    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_3d_vtkm_cuda_backend");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["runtime/backend"] = "cuda";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_multi_render)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }


    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");

    string output_path = prepare_output_dir();
    string image_prefix0 = "render_0";
    string output_file = conduit::utils::join_file_path(output_path,image_prefix0);

    // remove old images before rendering
    remove_test_image(output_file);

    string image_prefix1 = "render_1";
    string output_file1 = conduit::utils::join_file_path(output_path,image_prefix1);

    // remove old images before rendering
    remove_test_image(output_file1);


    //
    // Create the actions.
    //

    conduit::Node control_points;
    conduit::Node &point1 = control_points.append();
    point1["type"] = "rgb";
    point1["position"] = 0.;
    double color[3] = {1., 0., 0.};
    point1["color"].set_float64_ptr(color, 3);

    conduit::Node &point2 = control_points.append();
    point2["type"] = "rgb";
    point2["position"] = 0.5;
    color[0] = 0;
    color[1] = 1.;
    point2["color"].set_float64_ptr(color, 3);

    conduit::Node &point3 = control_points.append();
    point3["type"] = "rgb";
    point3["position"] = 1.0;
    color[0] = 1.;
    color[1] = 1.;
    color[2] = 1.;
    point3["color"].set_float64_ptr(color, 3);

    conduit::Node &point4 = control_points.append();
    point4["type"] = "alpha";
    point4["position"] = 0.;
    point4["alpha"] = 0.;

    conduit::Node &point5 = control_points.append();
    point5["type"] = "alpha";
    point5["position"] = 1.0;
    point5["alpha"] = 1.;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "volume";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/color_table/name"] = "blue";
    scenes["s1/plots/p1/color_table/control_points"] = control_points;

    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    //
    scenes["s1/renders/r2/image_width"]  = 400;
    scenes["s1/renders/r2/image_height"] = 400;
    scenes["s1/renders/r2/image_prefix"]   = output_file1;
    double vec3[3];
    vec3[0] = 1.; vec3[1] = 1.; vec3[2] = 1.;
    scenes["s1/renders/r2/camera/look_at"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = 25.; vec3[2] = 15.;
    scenes["s1/renders/r2/camera/position"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = -1.; vec3[2] = 0.;
    scenes["s1/renders/r2/camera/up"].set_float64_ptr(vec3,3);
    scenes["s1/renders/r2/camera/fov"] = 60.;
    scenes["s1/renders/r2/camera/xpan"] = 0.;
    scenes["s1/renders/r2/camera/ypan"] = 0.;
    scenes["s1/renders/r2/camera/zoom"] = 1.0;
    scenes["s1/renders/r2/camera/azimuth"] = 10.0;
    scenes["s1/renders/r2/camera/elevation"] = -10.0;
    scenes["s1/renders/r2/camera/near_plane"] = 0.1;
    scenes["s1/renders/r2/camera/far_plane"] = 100.1;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

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
    std::string msg = "An example of creating a render, specifying all camera parameters.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file1, 0.01f));
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_milk_chocolate)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");

    string output_path = prepare_output_dir();
    string image_prefix0 = "milk_chocolate";
    string output_file = conduit::utils::join_file_path(output_path,image_prefix0);

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node control_points;
    conduit::Node &point1 = control_points.append();
    point1["type"] = "rgb";
    point1["position"] = 0.;
    double color[3] = {.23, 0.08, 0.08};
    point1["color"].set_float64_ptr(color, 3);

    conduit::Node &point2 = control_points.append();
    point2["type"] = "rgb";
    point2["position"] = .5;
    color[0] = .48;
    color[1] = .23;
    color[2] = .04;
    point2["color"].set_float64_ptr(color, 3);

    conduit::Node &point3 = control_points.append();
    point3["type"] = "rgb";
    point3["position"] = 1.0;
    color[0] = .99;
    color[1] = 1.;
    color[2] = .96;
    point3["color"].set_float64_ptr(color, 3);

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/color_table/control_points"] = control_points;

    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

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
    std::string msg = "An example of creating a custom color map.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_compressed_color_table)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    ASCENT_INFO("Testing 3D Rendering with Compressed Color Table");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"vtkm_compressed_color_table");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node control_points;
    control_points["r"] = {.23, .48, .99};
    control_points["g"] = {0.08, .23, 1.};
    control_points["b"] = {0.08, .04, .96};
    control_points["a"] = {1., 1., 1.};
    control_points["position"] = {0., .5, 1.};

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/color_table/control_points"] = control_points;

    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

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
    std::string msg = "An example of creating a custom compressed color map.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_disable_color_bar)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));


    ASCENT_INFO("Testing 3D Rendering with Default Pipeline");

    string output_path = prepare_output_dir();
    string image_prefix0 = "no_color_bar";
    string output_file = conduit::utils::join_file_path(output_path,image_prefix0);

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/plots/p1/color_table/annotation"] = "false";

    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

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
    std::string msg = "An example of disabling a color table.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, render_3d_domain_overload)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with ascent support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D MPI "
                      "Runtime test");

        return;
    }


    Node multi_dom;
    Node &mesh1 = multi_dom.append();
    Node &mesh2 = multi_dom.append();
    //
    // Create the data.
    //
    Node verify_info;
    create_3d_example_dataset(mesh1,32,0,2);
    create_3d_example_dataset(mesh2,32,1,2);

    mesh1["state/domain_id"] = 0;
    mesh2["state/domain_id"] = 1;

    // There is a bug in conduit blueprint related to rectilinear
    // reenable this check after updating conduit
    // EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));
    conduit::blueprint::mesh::verify(multi_dom,verify_info);

    // make sure the _output dir exists
    string output_path = "";
    output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_domain_overload");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "rank_ele";
    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;
    scenes["s1/renders/r1/camera/azimuth"] = 45.0;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(multi_dom);
    ascent.execute(actions);
    ascent.close();
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of creating a render specifying the image size.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_render_3d, render_3d_empty_data)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with ascent support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D MPI "
                      "Runtime test");

        return;
    }

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "rank_ele";

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;


    //
    // Run Ascent
    //

    Ascent ascent;

    Node data; // empty!
    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    // we expect ascent to complain about no data
    EXPECT_THROW(ascent.execute(actions),conduit::Error);
    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_supported_field_dtypes)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }


    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              3,
                                              3,
                                              3,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering of fields with different data types");

    std::cout << std::endl;

    int num_vals = data["fields/braid/values"].dtype().number_of_elements();
    //
    // Create the actions.
    //

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    string output_path = prepare_output_dir();


    // ints

    // int 8
    {
        std::cout << "braid_int8" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int8");
        // remove old images before rendering
        remove_test_image(output_file);

        data["fields/braid/values"].set(DataType::int8(num_vals));
        int8_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }

        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // int 16
    {
        std::cout << "braid_int16" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int16");
        // remove old images before rendering
        remove_test_image(output_file);
        data["fields/braid/values"].set(DataType::int16(num_vals));
        int16_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }

    // int 32
    {
        std::cout << "braid_int32" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int32");
        // remove old images before rendering
        remove_test_image(output_file);

        data["fields/braid/values"].set(DataType::int32(num_vals));
        int32_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }

        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // int 64
    {
        std::cout << "braid_int64" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int64");
        // remove old images before rendering
        remove_test_image(output_file);
        data["fields/braid/values"].set(DataType::int64(num_vals));
        int64_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }


    // uints

    // uint 8
    {
        std::cout << "braid_uint8" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint8");
        // remove old images before rendering
        remove_test_image(output_file);

        data["fields/braid/values"].set(DataType::uint8(num_vals));
        uint8_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }

        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // uint 16
    {
        std::cout << "braid_uint16" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint16");
        // remove old images before rendering
        remove_test_image(output_file);
        data["fields/braid/values"].set(DataType::uint16(num_vals));
        uint16_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }

    // uint 32
    {
        std::cout << "braid_uint32" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint32");
        // remove old images before rendering
        remove_test_image(output_file);

        data["fields/braid/values"].set(DataType::uint32(num_vals));
        uint32_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }

        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // uint 64
    {
        std::cout << "braid_uint64" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint64");
        // remove old images before rendering
        remove_test_image(output_file);
        data["fields/braid/values"].set(DataType::uint64(num_vals));
        uint64_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }


    // fp types

    // float 32
    {
        std::cout << "braid_float32" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_float32");
        // remove old images before rendering
        remove_test_image(output_file);

        data["fields/braid/values"].set(DataType::float32(num_vals));
        float32_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }

    // float 64
    {
        std::cout << "braid_float64" << std::endl;
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_float64");
        // remove old images before rendering
        remove_test_image(output_file);

        data["fields/braid/values"].set(DataType::float64(num_vals));
        float64_array varray = data["fields/braid/values"].value();
        for(int i=0; i<num_vals; i++)
        {
            varray[i] = i;
        }
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }



    ascent.close();
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_supported_conn_dtypes)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }


    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              3,
                                              3,
                                              3,
                                              data);

    Node n_orig_coords = data["topologies/mesh/elements/connectivity"];

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering of fields with different data types");

    int num_vals = data["fields/braid/values"].dtype().number_of_elements();
    //
    // Create the actions.
    //

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    string output_path = prepare_output_dir();


    // ints

    // int 8
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int8_conn");
        // remove old images before rendering
        remove_test_image(output_file);

        n_orig_coords.to_int8_array(data["topologies/mesh/elements/connectivity"]);
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // int 16
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int16_conn");
        // remove old images before rendering
        remove_test_image(output_file);
        n_orig_coords.to_int16_array(data["topologies/mesh/elements/connectivity"]);
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }

    // int 32
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int32_conn");
        // remove old images before rendering
        remove_test_image(output_file);
        n_orig_coords.to_int32_array(data["topologies/mesh/elements/connectivity"]);
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // int 64
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_int64_conn");
        // remove old images before rendering
        remove_test_image(output_file);
        n_orig_coords.to_int64_array(data["topologies/mesh/elements/connectivity"]);
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }


    // uints

    // uint 8
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint8_conn");
        // remove old images before rendering
        remove_test_image(output_file);

        n_orig_coords.to_uint8_array(data["topologies/mesh/elements/connectivity"]);

        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // uint 16
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint16_conn");
        // remove old images before rendering
        remove_test_image(output_file);
        n_orig_coords.to_uint16_array(data["topologies/mesh/elements/connectivity"]);
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }

    // uint 32
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint32_conn");
        // remove old images before rendering
        remove_test_image(output_file);

        n_orig_coords.to_uint32_array(data["topologies/mesh/elements/connectivity"]);

        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }
    // uint 64
    {
        string output_file = conduit::utils::join_file_path(output_path,
                                        "tout_render_3d_braid_uint64_conn");
        // remove old images before rendering
        remove_test_image(output_file);
        n_orig_coords.to_uint64_array(data["topologies/mesh/elements/connectivity"]);
        ascent.publish(data);
        scenes["s1/image_prefix"] = output_file;
        ascent.execute(actions);
        // check that we created an image
        EXPECT_TRUE(check_test_image(output_file));
    }

    ascent.close();
}


//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_points_implicit_topo)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("points_implicit",
                                              10,
                                              10,
                                              10,
                                              data);


    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering of implicit points topology");

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";


    string output_path = prepare_output_dir();


    string output_file = conduit::utils::join_file_path(output_path,
                                    "tout_render_3d_braid_points_implicit");
    // remove old images before rendering
    remove_test_image(output_file);

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    scenes["s1/image_prefix"] = output_file;
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file,0.002));

}


//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_pyra)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }


    Node mesh, info;
    mesh["state/cycle"] = 100;
    // create the coordinate set
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"] = {-1.0,  1.0, 1.0, -1.0, 0.0};
    mesh["coordsets/coords/values/y"] = {-1.0, -1.0, 1.0,  1.0, 0.0};
    mesh["coordsets/coords/values/z"] = { 0.0,  0.0, 0.0,  0.0, 1.0};
    // add the topology

    mesh["topologies/topo/type"] = "unstructured";
    mesh["topologies/topo/coordset"] = "coords";
    mesh["topologies/topo/elements/shape"] = "pyramid";
    mesh["topologies/topo/elements/connectivity"].set(DataType::int64(5));
    int64_array con_vals = mesh["topologies/topo/elements/connectivity"].value();
    for(index_t i =0; i < 5; i++)
    {
      con_vals[i] = i;
    }

    mesh["fields/vert_id/topology"] = "topo";
    mesh["fields/vert_id/association"] = "vertex";
    mesh["fields/vert_id/values"].set(DataType::float64(5));
    float64_array vert_id_vals = mesh["fields/vert_id/values"].value();
    for(index_t i =0; i < 5; i++)
    {
      vert_id_vals[i] = i;
    }

    mesh["fields/ele_id/topology"] = "topo";
    mesh["fields/ele_id/association"] = "element";
    mesh["fields/ele_id/values"].set(DataType::float64(1));
    float64_array ele_id_vals = mesh["fields/ele_id/values"].value();
    ele_id_vals[0] = 0;

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                  "tout_render_3d_pyramid_vert_id");
    // remove old images before rendering
    remove_test_image(output_file);

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "vert_id";
    scenes["s1/image_prefix"] = output_file;


    Ascent ascent;
    ascent.open();
    ascent.publish(mesh);
    ascent.execute(actions);
    ascent.close();

    EXPECT_TRUE(check_test_image(output_file));

}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_wedge)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    Node mesh, info;
    mesh["state/cycle"] = 100;
    // create the coordinate set
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"] = {-1.0,  1.0, 0.0, -1.0, 1.0, 0.0};
    mesh["coordsets/coords/values/y"] = {-1.0, -1.0, -1.0,  1.0, 1.0, 1.0};
    mesh["coordsets/coords/values/z"] = { 0.0,  0.0, 1.0,  0.0, 0.0, 1.0};
    // add the topology

    mesh["topologies/topo/type"] = "unstructured";
    mesh["topologies/topo/coordset"] = "coords";
    mesh["topologies/topo/elements/shape"] = "wedge";
    mesh["topologies/topo/elements/connectivity"].set(DataType::int64(6));
    int64_array con_vals = mesh["topologies/topo/elements/connectivity"].value();
    for(index_t i =0; i < 6; i++)
    {
      con_vals[i] = i;
    }

    mesh["fields/vert_id/topology"] = "topo";
    mesh["fields/vert_id/association"] = "vertex";
    mesh["fields/vert_id/values"].set(DataType::float64(6));
    float64_array vert_id_vals = mesh["fields/vert_id/values"].value();
    for(index_t i =0; i < 6; i++)
    {
      vert_id_vals[i] = i;
    }

    mesh["fields/ele_id/topology"] = "topo";
    mesh["fields/ele_id/association"] = "element";
    mesh["fields/ele_id/values"].set(DataType::float64(1));
    float64_array ele_id_vals = mesh["fields/ele_id/values"].value();
    ele_id_vals[0] = 0;

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                  "tout_render_3d_wedge_vert_id");
    // remove old images before rendering
    remove_test_image(output_file);

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "vert_id";
    scenes["s1/image_prefix"] = output_file;


    Ascent ascent;
    ascent.open();
    ascent.publish(mesh);
    ascent.execute(actions);
    ascent.close();

    EXPECT_TRUE(check_test_image(output_file));

}



// //-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_extreme_extents)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    // create uniform grid with very large (spatial) extents
    Node mesh, info;

    int num_per_dim = 9;
    double total_side_dist = 1.5e20;
    mesh["state/cycle"] = 100;
    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = num_per_dim;
    mesh["coordsets/coords/dims/j"] = num_per_dim;
    mesh["coordsets/coords/dims/k"] = num_per_dim;

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"].set_float64(0.0);
    mesh["coordsets/coords/origin/y"].set_float64(0.0);
    mesh["coordsets/coords/origin/z"].set_float64(0.0);

    double distance_per_step = total_side_dist/(double)(num_per_dim-1);
    mesh["coordsets/coords/spacing/dx"] = distance_per_step;
    mesh["coordsets/coords/spacing/dy"] = distance_per_step;
    mesh["coordsets/coords/spacing/dz"] = distance_per_step;

    // add the topology
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/topo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/topo/coordset"] = "coords";

    int num_verts = num_per_dim * num_per_dim * num_per_dim; // 3D
    mesh["fields/alternating/values"].set(DataType::float64(num_verts));

    // create a vertex associated field named alternating
    mesh["fields/alternating/association"] = "vertex";
    mesh["fields/alternating/topology"] = "topo";
    float64_array vals = mesh["fields/alternating/values"].value();
    for (int i = 0 ; i < num_verts ; i++)
     vals[i] = ( (i%2)==0 ? 0.0 : 1.0);

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "alternating";


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                  "tout_render_3d_extreme_extents");
    // remove old images before rendering
    remove_test_image(output_file);

    Ascent ascent;
    ascent.open();
    ascent.publish(mesh);

    scenes["s1/image_prefix"] = output_file;
    ascent.execute(actions);
    // check that we created an image
    // TODO: We expect this to fail until we address float64 vs float32 issues
    EXPECT_FALSE(check_test_image(output_file));

    //now with rectilinear:
    Node mesh_recti_coords;
    conduit::blueprint::mesh::coordset::uniform::to_rectilinear(mesh["coordsets/coords"],
                                                                mesh_recti_coords);
    mesh_recti_coords.print();

    mesh["coordsets/coords"] = mesh_recti_coords;
    mesh["topologies/topo/type"]  = "rectilinear";
    mesh.print();

    bool mesh_ok = conduit::blueprint::mesh::verify(mesh,info);
    EXPECT_TRUE(mesh_ok);
    if(!mesh_ok)
    {
        std::cout << "Mesh Blueprint Verify Failed!! " << info.to_yaml() << std::endl;
    }

    output_file = conduit::utils::join_file_path(output_path,
                                  "tout_render_3d_extreme_recti_extents");
    // remove old images before rendering
    remove_test_image(output_file);

    ascent.publish(mesh);
    scenes["s1/image_prefix"] = output_file;
    ascent.execute(actions);

    // check that we created an image
    // TODO: We expect this to fail until we address float64 vs float32 issues
    EXPECT_FALSE(check_test_image(output_file));


    //now with unstructured:
    Node mesh_unstruct;

    conduit::blueprint::mesh::topology::rectilinear::to_unstructured(mesh["topologies/topo"],
                                                                     mesh_unstruct["topologies/topo"],
                                                                     mesh_unstruct["coordsets/coords"]);

    mesh["coordsets/coords"]  = mesh_unstruct["coordsets/coords"];
    mesh["topologies/topo/"]  = mesh_unstruct["topologies/topo"];
    mesh.print();

    mesh_ok = conduit::blueprint::mesh::verify(mesh,info);
    EXPECT_TRUE(mesh_ok);
    if(!mesh_ok)
    {
        std::cout << "Mesh Blueprint Verify Failed!! " << info.to_yaml() << std::endl;
    }

    output_file = conduit::utils::join_file_path(output_path,
                                  "tout_render_3d_extreme_unstruct_extents");
    // remove old images before rendering
    remove_test_image(output_file);

    ascent.publish(mesh);
    scenes["s1/image_prefix"] = output_file;
    ascent.execute(actions);

    // check that we created an image
    // TODO: We expect this to fail until we address float64 vs float32 issues
    EXPECT_FALSE(check_test_image(output_file));


}



// //-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_ray_zcull_bug)
{
    //
    // this test exercises a ray culling bug in vtk-m
    // the bug improperly culls rays that are valid due to a bad
    // check related to the near and far plane
    // we have a patch and will work on getting a fixed in to 2.2
    //

    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    // create a tiny mesh that will be far away from the camera
    Node mesh;
    mesh["state/cycle"] = 100;
    int num_per_dim = 6;
    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = num_per_dim;
    mesh["coordsets/coords/dims/j"] = num_per_dim;
    mesh["coordsets/coords/dims/k"] = num_per_dim;

    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"].set_float64(-3.0);
    mesh["coordsets/coords/origin/y"].set_float64(-3.0);
    mesh["coordsets/coords/origin/z"].set_float64(10000.0);

    mesh["coordsets/coords/spacing/dx"] = 1.0;
    mesh["coordsets/coords/spacing/dy"] = 1.0;
    mesh["coordsets/coords/spacing/dz"] = 1.0;

    // add the topology
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/topo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/topo/coordset"] = "coords";

    int num_verts = num_per_dim * num_per_dim * num_per_dim; // 3D
    // create a vertex associated field named alternating
    mesh["fields/vert_ids/association"] = "vertex";
    mesh["fields/vert_ids/topology"] = "topo";
    mesh["fields/vert_ids/values"].set(DataType::float64(num_verts));
    float64_array vals_vert_ids =  mesh["fields/vert_ids/values"].value();

    for(index_t i=0;i< num_verts;i++)
    {
        vals_vert_ids[i] = i;
    }

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                  "tout_render_3d_ray_z_cull_bug");
    // remove old images before rendering
    remove_test_image(output_file);

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "vert_ids";

    Node &r1 = scenes["s1/renders/r1"];

    std::string cam_yaml = R"INLINE(
world_annotations: false
camera:
  fov: 30.0
  look_at: [0.0,0.0,10000.0]
  position: [0.0,0.0,0.0]
  up: [0.0, 1.0, 0.0]
  zoom: [100]
  near_plane: 9000.0
  far_plane: 11000.0
)INLINE";
    r1.parse(cam_yaml,"yaml");

    r1["image_prefix"] = output_file;
    std::cout << actions.to_yaml() << std::endl;
    Ascent ascent;
    ascent.open();
    ascent.publish(mesh);
    ascent.execute(actions);
    // check that we created an image
    EXPECT_TRUE(check_test_image(output_file));

}

TEST(ascent_render_3d, test_render_3d_camera_frustum_meshes)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);

    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {

        ASCENT_INFO("Ascent support disabled, skipping camera frustum mesh tests");

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

    ASCENT_INFO("Testing 3D rendering of camera frustum meshes\n");

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";

    add_plots["scenes/s1/plots/p1/type"] = "pseudocolor";
    add_plots["scenes/s1/plots/p1/field"] = "braid";

    string output_path = prepare_output_dir();

    // Base Case
    // Case to verify that the frustrums plot correctly without rotations.
    add_plots["scenes/s1/renders/r1/image_prefix"] =
        conduit::utils::join_file_path(output_path, "tout_render_3d_frust_image_0_az_0_el_0_");
    add_plots["scenes/s1/renders/r1/camera/azimuth"] = 0.0;
    add_plots["scenes/s1/renders/r1/camera/elevation"] = 0.0;

    // Azimuth Check
    // Case to verify that when only the azimuth is changes the frustum is plotted appropriately.
    // Additionally verifies that the up vector is not changed and still points directly up.
    add_plots["scenes/s1/renders/r2/image_prefix"] =
        conduit::utils::join_file_path(output_path, "tout_render_3d_frust_image_1_az_120_el_0_");
    add_plots["scenes/s1/renders/r2/camera/azimuth"] = 120.0;
    add_plots["scenes/s1/renders/r2/camera/elevation"] = 0.0;

    // Elevation Check
    // Case to verify that the frustum is plotted correctly when only elevation changes.
    // Additionally checks that for angles over 90 degrees that the up vector will angle downwards.
    add_plots["scenes/s1/renders/r3/image_prefix"] =
        conduit::utils::join_file_path(output_path, "tout_render_3d_frust_image_2_az_0_el_120_");
    add_plots["scenes/s1/renders/r3/camera/azimuth"] = 0.0;
    add_plots["scenes/s1/renders/r3/camera/elevation"] = 120;

    // Mixed Rotation Check
    // Verify that when both the azimuth and elevation have been changed that the frustum is correct.
    add_plots["scenes/s1/renders/r4/image_prefix"] =
        conduit::utils::join_file_path(output_path, "tout_render_3d_frust_image_3_az_20_el_-45_");
    add_plots["scenes/s1/renders/r4/camera/azimuth"] = 20.0;
    add_plots["scenes/s1/renders/r4/camera/elevation"] = -45.0;

    // Test that if the look_at location and the position location are quite similar that the frustum
    // is still generated without errors.
    add_plots["scenes/s1/renders/r5/image_prefix"] =
        conduit::utils::join_file_path(output_path, "tout_render_3d_frust_image_nan");
    add_plots["scenes/s1/renders/r5/camera/position"] = {0.0, 0.0, 0.0581200011074543};
    add_plots["scenes/s1/renders/r5/camera/look_at"] = {0.0, 0.0, 0.0};

    //
    // Run Ascent to generate images
    //
    Ascent ascent;

    Node ascent_opts, ascent_info;
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(ascent_info);
    ascent.close();

    //
    // For each image that was generated, run ascent to visualize the camera frustum
    //
    for (int image_index = 0; image_index<5; image_index++) {
        conduit::Node &image_node = ascent_info["images"][image_index];
        conduit::Node camera_data = image_node["camera/camera_frustum_mesh"];

        string image_name_root = conduit::utils::join_file_path(output_path,
            "tout_render_3d_frust_camera_image_" + std::to_string(image_index));
        //conduit::relay::io::blueprint::save_mesh(camera_data, image_name_root + "_frustum_mesh","hdf5");

        Ascent ascent_2;
        ascent_2.open();
        ascent_2.publish(camera_data);

        conduit::Node frustum_actions;
        conduit::Node &add_frustum_plots = frustum_actions.append();
        add_frustum_plots["action"] = "add_scenes";
        add_frustum_plots["scenes/s1/plots/p1/type"] = "mesh";
        add_frustum_plots["scenes/s1/plots/p1/topology"] = "camera_frustum_topo";
        add_frustum_plots["scenes/s1/plots/p2/type"] = "mesh";
        add_frustum_plots["scenes/s1/plots/p2/topology"] = "clipping_planes_topo";
        add_frustum_plots["scenes/s1/plots/p3/type"] = "mesh";
        add_frustum_plots["scenes/s1/plots/p3/topology"] = "scene_bounds_topo";

        // Render a plot of the camera frustum to verify it's relation to the scene
        std::string frust_plot_file_1 = image_name_root + "_frustum_front_image_";
        remove_test_image(frust_plot_file_1);
        add_frustum_plots["scenes/s1/renders/r1/image_prefix"] = frust_plot_file_1;
        add_frustum_plots["scenes/s1/renders/r1/camera/azimuth"] = 0.0;
        add_frustum_plots["scenes/s1/renders/r1/camera/elevation"] = 0.0;
        add_frustum_plots["scenes/s1/renders/r1/annotations"] = "false";

        // Render a plot of the camera frustum at a 90 degree angle to see the frustum better
        std::string frust_plot_file_2 = image_name_root + "_frustum_side_image_";
        remove_test_image(frust_plot_file_2);
        add_frustum_plots["scenes/s1/renders/r2/image_prefix"] = frust_plot_file_2;
        add_frustum_plots["scenes/s1/renders/r2/camera/azimuth"] = 90.0;
        add_frustum_plots["scenes/s1/renders/r2/camera/elevation"] = 0.0;
        add_frustum_plots["scenes/s1/renders/r2/annotations"] = "false";

        ascent_2.execute(frustum_actions);
        ascent_2.close();

        // check that we created an image
        EXPECT_TRUE(check_test_image(frust_plot_file_1));
        EXPECT_TRUE(check_test_image(frust_plot_file_2));
    }
}
//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_bentgrid_example)
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
    conduit::relay::io::blueprint::read_mesh(test_data_file("bentgrid_3d_visitghost_yaml.root"),
                                             data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_render_3d_bentgrid");
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
TEST(ascent_render_3d, test_render_3d_zero_zoom_handled)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

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

    ASCENT_INFO("Testing Error Handling of Render Camera Zoom Parameter\n");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_render_3d_zero_zoom_handled");

    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";
    scenes["s1/image_prefix"] = output_file;
    scenes["s1/renders/r1/image_prefix"]   = output_file;
    scenes["s1/renders/r1/camera/zoom"] = 0.0;


    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["timings"] = "true";
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);

    bool error = false;
    try
    {
        ascent.execute(actions);
    }
    catch(...)
    {
        error = true;
    }

    ascent.close();

    EXPECT_TRUE(error);
}

//-----------------------------------------------------------------------------
TEST(ascent_render_3d, test_render_3d_multi_topo_extents)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    std::string data_yml = R"xyzxyz(
        state:
            cycle: 100
        coordsets: 
          coords_small: 
            type: "uniform"
            dims: 
              i: 3
              j: 3
              k: 3
            origin: 
              x: -10.0
              y: -10.0
              z: -10.0
            spacing: 
              dx: 10.0
              dy: 10.0
              dz: 10.0
          coords_big: 
            type: "uniform"
            dims: 
              i: 3
              j: 3
              k: 3
            origin: 
              x: -1000.0
              y: -1000.0
              z: -1000.0
            spacing: 
              dx: 1000.0
              dy: 1000.0
              dz: 1000.0
        topologies: 
          topo_small: 
            type: "uniform"
            coordset: "coords_small"
          topo_big: 
            type: "uniform"
            coordset: "coords_big"
        fields: 
          field_small: 
            association: "element"
            topology: "topo_small"
            values: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
          field_big: 
            association: "element"
            topology: "topo_big"
            values: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    )xyzxyz";

    Node data, verify_info;
    data.parse(data_yml,"yaml");
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with multi topo with varying extends");

    string output_path = prepare_output_dir();
    string output_base = conduit::utils::join_file_path(output_path,"tout_render_3d_multi_topo_extents");
    string output_fsmall = output_base + "_small";
    string output_fbig   = output_base + "_big";

    // remove old images before rendering
    remove_test_image(output_fsmall);
    remove_test_image(output_fbig);

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";

    conduit::Node & scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "field_small";
    scenes["s1/image_prefix"] = output_fsmall;

    scenes["s2/plots/p1/type"]  = "pseudocolor";
    scenes["s2/plots/p1/field"] = "field_big";
    scenes["s2/image_prefix"] = output_fbig;

    scenes["s2/plots/p2/type"]  = "pseudocolor";
    scenes["s2/plots/p2/field"] = "field_big";
    scenes["s2/image_prefix"] = output_fbig;


    scenes["s2/plots/p3/type"]  = "pseudocolor";
    scenes["s2/plots/p3/field"] = "field_big";
    scenes["s2/image_prefix"] = output_fbig;


    Node info;
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(info);
    ascent.close();

    ofstream ofs;
    ofs.open(output_base + "_graph.html");
    ofs <<info["flow_graph_dot_html"].as_string();
    ofs.close();

    EXPECT_TRUE(check_test_image(output_fsmall));
    EXPECT_TRUE(check_test_image(output_fbig));
}

//-----------------------------------------------------------------------------

TEST(ascent_render_3d, test_render_ascent_camera)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with an ascent camera\n");

    string output_path = prepare_output_dir();
    string image_prefix = "t_out_render_3d_ascent_camera";
    string output_file = conduit::utils::join_file_path(output_path,image_prefix);

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "pseudocolor";
    scenes["s1/plots/p1/field"] = "braid";

    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    // set the camera parameters
    double vec3[3];
    // vec3[0] = 1.; vec3[1] = 1.; vec3[2] = 1.;
    vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 0.;
    scenes["s1/renders/r1/camera/look_at"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = 25.; vec3[2] = 25.;
    // vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 0.;
    scenes["s1/renders/r1/camera/position"].set_float64_ptr(vec3,3);
    // vec3[0] = 0.; vec3[1] = -1.; vec3[2] = 0.;
    // scenes["s1/renders/r1/camera/up"].set_float64_ptr(vec3,3);
    // scenes["s1/renders/r1/camera/fov"] = 60.;
    // scenes["s1/renders/r1/camera/xpan"] = 0.;
    // scenes["s1/renders/r1/camera/ypan"] = 0.;
    // scenes["s1/renders/r1/camera/zoom"] = 1.0;
    // scenes["s1/renders/r1/camera/azimuth"] = 10.0;
    // scenes["s1/renders/r1/camera/elevation"] = -10.0;
    scenes["s1/renders/r1/camera/near_plane"] = 26.;
    // scenes["s1/renders/r1/camera/far_plane"] = 100.1;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    actions.print();

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
    std::string msg = "An example of creating a render, specifying all camera parameters.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------

TEST(ascent_render_3d, test_render_visit_camera)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with an ascent camera\n");

    string output_path = prepare_output_dir();
    string image_prefix = "t_out_render_3d_visit_camera";
    string output_file = conduit::utils::join_file_path(output_path,image_prefix);

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "volume";
    scenes["s1/plots/p1/field"] = "braid";

    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    // set the camera parameters
    double vec3[3];
    vec3[0] = 0.8; vec3[1] = .5; vec3[2] = .5;
    scenes["s1/renders/r1/camera/viewNormal"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 0.;
    scenes["s1/renders/r1/camera/focus"].set_float64_ptr(vec3,3);
    vec3[0] = .5; vec3[1] = 0.; vec3[2] = .5;
    scenes["s1/renders/r1/camera/viewUp"].set_float64_ptr(vec3,3);
    scenes["s1/renders/r1/camera/viewAngle"] = 30.;
    scenes["s1/renders/r1/camera/parallelScale"] = 30.;
    scenes["s1/renders/r1/camera/nearPlane"] = 10.;
    scenes["s1/renders/r1/camera/farPlane"] = 11.;
    double vec2[2];
    vec2[0] = 0.; vec2[1] = 0.;
    scenes["s1/renders/r1/camera/imagePan"].set_float64_ptr(vec2,2);
    scenes["s1/renders/r1/camera/imageZoom"] = 1;
    // scenes["s1/renders/r1/camera/perspective"] = 1.; // true
    // scenes["s1/renders/r1/camera/eyeAngle"] = 2.;
    // scenes["s1/renders/r1/camera/centerOfRotationSet"] = 0.; // false
    // vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 0.;
    // scenes["s1/renders/r1/camera/centerOfRotation"].set_float64_ptr(vec3,3);
    // scenes["s1/renders/r1/camera/axis3DScaleFlag"] = 0.; // false
    // vec3[0] = 1.; vec3[1] = 1.; vec3[2] = 1.;
    // scenes["s1/renders/r1/camera/axis3DScale"].set_float64_ptr(vec3,3);
    // vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 1.;
    // scenes["s1/renders/r1/camera/shear"].set_float64_ptr(vec3,3);
    // scenes["s1/renders/r1/camera/windowValid"] = 0.; // false

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    conduit::Node &add_extract = actions.append();
    add_extract["action"] = "add_extracts";
    add_extract["extracts/e1/type"] = "relay";
    add_extract["extracts/e1/params/path"] = "out_extract_braid_example";
    add_extract["extracts/e1/params/protocol"] = "blueprint/mesh/hdf5";

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
    std::string msg = "An example of creating a render, specifying all camera parameters.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------

TEST(ascent_render_3d, test_render_invalid_camera)
{
    // the ascent runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping 3D default"
                      "Pipeline test");

        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing 3D Rendering with an ascent camera\n");

    string output_path = prepare_output_dir();
    string image_prefix = "t_out_render_3d_mixed_camera";
    string output_file = conduit::utils::join_file_path(output_path,image_prefix);

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]  = "volume";
    scenes["s1/plots/p1/field"] = "braid";

    scenes["s1/image_prefix"] = output_file;

    scenes["s1/renders/r1/image_width"]  = 512;
    scenes["s1/renders/r1/image_height"] = 512;
    scenes["s1/renders/r1/image_prefix"]   = output_file;

    // set the camera parameters (this is an INVALID set of variables mixing the two input formats)
    double vec3[3];
    vec3[0] = 1.; vec3[1] = 1.; vec3[2] = 1.;
    scenes["s1/renders/r1/camera/look_at"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = 25.; vec3[2] = 25.;
    scenes["s1/renders/r1/camera/position"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = -1.; vec3[2] = 0.;
    scenes["s1/renders/r1/camera/xpan"] = 0.;
    scenes["s1/renders/r1/camera/ypan"] = 0.;
    scenes["s1/renders/r1/camera/azimuth"] = 10.0;
    scenes["s1/renders/r1/camera/elevation"] = -10.0;
    scenes["s1/renders/r1/camera/far_plane"] = 100.1;
    vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 1.;
    scenes["s1/renders/r1/camera/viewNormal"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 0.;
    scenes["s1/renders/r1/camera/focus"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = 1.; vec3[2] = 0.;
    scenes["s1/renders/r1/camera/viewUp"].set_float64_ptr(vec3,3);
    scenes["s1/renders/r1/camera/viewAngle"] = 30.;
    scenes["s1/renders/r1/camera/parallelScale"] = 1.;
    scenes["s1/renders/r1/camera/nearPlane"] = 0.001;
    scenes["s1/renders/r1/camera/imageZoom"] = 1.0;
    scenes["s1/renders/r1/camera/perspective"] = 1.; // true
    scenes["s1/renders/r1/camera/eyeAngle"] = 2.;
    scenes["s1/renders/r1/camera/centerOfRotationSet"] = 0.; // false
    vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 0.;
    scenes["s1/renders/r1/camera/centerOfRotation"].set_float64_ptr(vec3,3);
    scenes["s1/renders/r1/camera/axis3DScaleFlag"] = 0.; // false
    vec3[0] = 1.; vec3[1] = 1.; vec3[2] = 1.;
    scenes["s1/renders/r1/camera/axis3DScale"].set_float64_ptr(vec3,3);
    vec3[0] = 0.; vec3[1] = 0.; vec3[2] = 1.;
    scenes["s1/renders/r1/camera/shear"].set_float64_ptr(vec3,3);
    scenes["s1/renders/r1/camera/windowValid"] = 0.; // false

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

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
    std::string msg = "An example of creating a render, specifying all camera parameters.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

// View3DAttributes 
    // viewNormal = (0, 0, 1)        -> The direction the camera is looking (from camera to focus point)
    // focus = (0, 0, 0)             -> The center point the camera is looking at
    // viewUp = (0, 1, 0)            -> The up direction relative to the camera's orientation
    // viewAngle = 30                -> Field of view (in degrees) from perspective projection
    // // parallelScale = 1             -> Zoom for orthographic projection
    // nearPlane = 0.001             -> Near clipping plane
    // farPlane = 100                -> Far clipping plane
    // // imagePan = (0, 0)             -> Pan offset applied in screen space
    // imageZoom = 1                 -> Additional zoom applied to the view
    // perspective = true            -> True for perspective and false for orthographic
    // // eyeAngle = 2                  -> Eye seperation angle in degrees for stereo rendering
    // // centerOfRotationSet = false   -> Indicates if an explicit center of rotation was set by the user
    // // centerOfRotation = (0, 0, 0)  -> The point around which 3D rotations will pivot
    // // axis3DScaleFlag = false       -> Boolean indicating if we should apply non-uniform scaling to the 3 axes
    // // axis3DScale = (1, 1, 1)       -> Axis scaling factors for each axis
    // // shear = (0, 0, 1)             -> For oblique projections or special visualization effects
    // // windowValid = false           -> 
// https://github.com/visit-dav/visit/blob/e54f4b532291b3e384ad2d7f27e810a41a6a3ca3/src/java/View3DAttributes.java#L23

// Ascent Camera:
    // look_at                       -> The target point that the camera is looking towards
    // position                      -> Camera position in 3D
    // up                            -> Up vector
    // fov                           -> Field of view in degrees
    // xpan                          -> Pan offset applied in screen space - x cooredinate
    // ypan                          -> Pan offset applied in screen space - y cooredinate
    // elevation                     -> 
    // azimuth
    // zoom                          -> Additional zoom applied to the view
    // near_plane                    -> Near clipping plane
    // far_plane                     -> Far clipping plane

// View2DAttributes 
    // windowCoords = (0, 0, 1, 1)
    // viewportCoords = (0.1, 0.1, 0.9, 0.9)
    // fullFrameActivationMode = Auto?
    // fullFrameAutoThreshold = 100
    // xScale = 0
    // yScale = 0
    // windowValid = false
// https://github.com/visit-dav/visit/blob/e54f4b532291b3e384ad2d7f27e810a41a6a3ca3/src/common/state/View3DAttributes.C

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


