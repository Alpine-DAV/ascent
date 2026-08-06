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
TEST(ascent_scalar_rendering, test_scalar_rendering)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
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

    ASCENT_INFO("Testing Scalar Rendering");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["image_width"] = 512;
    params["image_height"] = 512;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_contour)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
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

    ASCENT_INFO("Testing Scalar Rendering of a Contour");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_contour");

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
    
    pipelines["pl1/f2/type"] = "project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f2/params"];
    params["image_width"] = 512;
    params["image_height"] = 512;

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering of a contour";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_fields_specified)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
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

    ASCENT_INFO("Testing Scalar Rendering with fields specified");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_fields_specified");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["image_width"] = 512;
    params["image_height"] = 512;
    params["fields"].append() = "braid";

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering of specific fields";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}



//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_data_bounds_specified)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
        return;
    }


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

    ASCENT_INFO("Testing Scalar Rendering with fields specified");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_bounds_specified");

    conduit::relay::io::blueprint::save_mesh(data,conduit::utils::join_file_path(output_path,
                                                "tout_scalar_rendering_bounds_specified_input"),"hdf5");

    //
    // Create the actions.
    //

    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["image_width"] = 512;
    params["image_height"] = 512;
    params["dataset_bounds"] = {  0.0, 5.0, // x
                                -10.0,-5.0, // y
                                  0.0,0.0}; // z

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering of specific fields";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_2d_camera)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
        return;
    }


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

    ASCENT_INFO("Testing Scalar Rendering with a 2d camera");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_2d_camera");

    conduit::relay::io::blueprint::save_mesh(data,conduit::utils::join_file_path(output_path,
                                                  "tout_scalar_rendering_2d_camera_input"),"hdf5");

    //
    // Create the actions.
    //

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "project_2d";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["image_width"]  = 512;
    params["image_height"] = 512;
    params["camera/2d"] = { -7.0, 3.0, 0.0,4.0 };

    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts=add_extracts["extracts"];;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";


    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering with 2d camera mode";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_field_filtering)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
        return;
    }


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

    ASCENT_INFO("Testing Scalar Rendering with a 2d camera");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_2d_field_filtering");

    conduit::relay::io::blueprint::save_mesh(data,conduit::utils::join_file_path(output_path,
                                                  "tout_scalar_rendering_2d_field_filtering_input"),"hdf5");

    //
    // Create the actions.
    //

    conduit::Node actions;
    conduit::Node &declare_fields= actions.append();
    declare_fields["action"] = "declare_fields";
    declare_fields["fields"].append() = "braid";
    declare_fields["fields"].append() = "radial";

    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    conduit::Node &pipelines = add_pipelines["pipelines"];
    pipelines["pl1/f1/type"] = "project_2d";
    conduit::Node &params = pipelines["pl1/f1/params"];
    params["image_width"]  = 512;
    params["image_height"] = 512;
    params["camera/2d"] = { -7.0, 3.0, 0.0,4.0 };

    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts=add_extracts["extracts"];;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";
    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";


    Ascent ascent;
    conduit::Node opts;
    opts["field_filtering"] = "true";
    ascent.open(opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering with filter filtering";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_contour_rays_output)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
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

    ASCENT_INFO("Testing Scalar Rendering of a Contour with Rays Output");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_contour_rays");

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
    
    pipelines["pl1/f2/type"] = "project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f2/params"];
    params["image_width"] = 512;
    params["image_height"] = 512;
    params["result"] = "rays";

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering of a contour";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_scalar_rendering, test_scalar_rendering_contour_explicit_rays_with_rays_output)
{
    // the viskores runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent support disabled, skipping test");
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

    ASCENT_INFO("Testing Scalar Rendering of a Contour Explicit Rays with Rays Output");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_scalar_rendering_explicit_rays");

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
    
    pipelines["pl1/f2/type"] = "project_2d";
    // filter knobs
    conduit::Node &params = pipelines["pl1/f2/params"];
    
    params["rays/points"]  = { 0.0, 0.0, 0.0, // center
                              -2.5, 0.0, 0.0, // left
                               2.5, 0.0, 0.0, // right
                               0.0, -2.5, 0.0,// bottom
                               0.0, 2.5, 0.0};  // top

    params["rays/normals"] = {0.0, 0.0, 1.0,
                              0.0, 0.0, 1.0,
                              0.0, 0.0, 1.0,
                              0.0, 0.0, 1.0};

    params["rays/max_distance"] = 20.0;
    params["result"] = "rays";

    conduit::Node extracts;
    extracts["e1/type"]  = "relay";
    extracts["e1/pipeline"] = "pl1";

    extracts["e1/params/path"] = output_file;
    extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

    conduit::Node actions;
    // add the extracts
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
    // add the pipeline
    conduit::Node &add_pipelines= actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    std::string msg = "An example of scalar rendering of a contour";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
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


