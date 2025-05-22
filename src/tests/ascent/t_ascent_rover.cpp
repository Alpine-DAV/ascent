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

constexpr index_t EXAMPLE_MESH_SIDE_DIM = 20;

//
// Utilities
//

void render_blueprint_result(const string &field_name,
                             const string &output_file,
                             const Node &data)
{
    //
    // Create the actions.
    //

    Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = field_name;
    scenes["s1/renders/r1/image_prefix"] = output_file;
    // TODO: Revisit this once we figure out why/where the data is rotated in the first place
    scenes["s1/renders/r1/camera/azimuth"] = 90.0;

    Node actions;
    Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    // TODO can we ask Ascent for the name of the file it wrote?
    // std::cout << ascent.info().to_yaml() << std::endl;
    ascent.close();
}

bool is_vtkm_disabled(const Node &about)
{
    bool vtk_disabled = about["runtimes/ascent/vtkm/status"].as_string() == "disabled";
    if (vtk_disabled)
    {
        ASCENT_INFO("vtkm support disabled, skipping test\n");
    }
    return vtk_disabled;
}

void get_valid_test_data(Node &data)
{
    Node verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
}

//
// Rover X-Ray tests
//

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtkm_disabled(n))
    {
        return;
    }

    //
    // Create an example mesh.
    //
    
    Node data;
    get_valid_test_data(data);

    ASCENT_INFO("Testing xray_extract on conduit braid example\n");

    const std::string query_output_path = prepare_output_dir();
    const std::string query_output_file = 
        conduit::utils::join_file_path(query_output_path, "tout_rover_xray_query");

    // remove old images before rendering
    remove_test_image(query_output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/absorption"] = "radial";
    extracts["e1/params/emission"] = "radial";
    extracts["e1/params/filename"] = query_output_file;
    extracts["e1/params/blueprint"] = "json";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    const std::string full_outfile_name = query_output_file + "_000100.cycle_000100.root";

    Node load_mesh, verify_info;
    conduit::relay::io::blueprint::load_mesh(full_outfile_name, load_mesh);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(load_mesh, verify_info));

    const std::string image_output_path = prepare_output_dir();
    const std::string image_output_base =
        conduit::utils::join_file_path(image_output_path, "tout_rover_xray_blueprint");

    render_blueprint_result("intensities", image_output_base, load_mesh);

    // TODO we need the render to make an interesting picture. This will be accomplished
    // by working on the basic mesh output and changing the order of the dimensions.
    // TODO we will need to change the baseline when we are making good renders.
    EXPECT_TRUE(check_test_image(image_output_base));

    std::string msg = "TODO we need a good description here";
    ASCENT_ACTIONS_DUMP(actions, image_output_base, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_curv3d)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtkm_disabled(n))
    {
        return;
    }

    //
    // Open an example mesh.
    //
    Node data, verify_info;
    const std::string root_file = 
        conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),
                                       "curv3d_blueprint.cycle_000048.root");

    conduit::relay::io::blueprint::load_mesh(root_file, data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    ASCENT_INFO("Testing xray_extract on curv3d example\n");

    const std::string query_output_path = prepare_output_dir();
    const std::string query_output_file = 
        conduit::utils::join_file_path(query_output_path,
                                       "tout_rover_xray_curv3d_blueprint_query");

    // remove old images before rendering
    remove_test_image(query_output_file, 48);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/absorption"] = "d";
    extracts["e1/params/emission"] = "p";
    extracts["e1/params/filename"] = query_output_file;
    extracts["e1/params/blueprint"] = "json";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    // TODO can we ask Ascent for the name of the file it wrote?
    // std::cout << ascent.info().to_yaml() << std::endl;
    ascent.close();

    const std::string full_outfile_name = query_output_file + "_000048.cycle_000048.root";

    Node load_mesh;
    conduit::relay::io::blueprint::load_mesh(full_outfile_name, load_mesh);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(load_mesh, verify_info));

    const std::string image_output_path = prepare_output_dir();
    const std::string image_output_base =
        conduit::utils::join_file_path(image_output_path, "tout_rover_xray_curv3d");

    render_blueprint_result("intensities", image_output_base, load_mesh);

    // TODO we need the render to make an interesting picture. This will be accomplished
    // by working on the basic mesh output and changing the order of the dimensions.
    // TODO we will need to change the baseline when we are making good renders.
    EXPECT_TRUE(check_test_image(image_output_base, 0.01f, 48));
    
    std::string msg = "TODO we need a good description here";
    ASCENT_ACTIONS_DUMP(actions, image_output_base, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_multi_curv3d)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtkm_disabled(n))
    {
        return;
    }

    //
    // Open an example mesh.
    //
    Node data, verify_info;
    const std::string root_file = 
        conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),
                                       "multi_curv3d_blueprint.cycle_000048.root");

    conduit::relay::io::blueprint::load_mesh(root_file, data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    ASCENT_INFO("Testing xray_extract on multi_curv3d example\n");

    const std::string query_output_path = prepare_output_dir();
    const std::string query_output_file = 
        conduit::utils::join_file_path(query_output_path,
                                       "tout_rover_xray_multi_curv3d_blueprint_query");

    // remove old images before rendering
    remove_test_image(query_output_file, 48);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/absorption"] = "d";
    extracts["e1/params/emission"] = "p";
    extracts["e1/params/filename"] = query_output_file;
    extracts["e1/params/blueprint"] = "json";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    // TODO can we ask Ascent for the name of the file it wrote?
    // std::cout << ascent.info().to_yaml() << std::endl;
    ascent.close();

    const std::string full_outfile_name = query_output_file + "_000048.cycle_000048.root";

    Node load_mesh;
    conduit::relay::io::blueprint::load_mesh(full_outfile_name, load_mesh);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(load_mesh, verify_info));

    const std::string image_output_path = prepare_output_dir();
    const std::string image_output_base =
        conduit::utils::join_file_path(image_output_path, "tout_rover_xray_multi_curv3d");

    render_blueprint_result("intensities", image_output_base, load_mesh);

    // TODO we need the render to make an interesting picture. This will be accomplished
    // by working on the basic mesh output and changing the order of the dimensions.
    // TODO we will need to change the baseline when we are making good renders.
    EXPECT_TRUE(check_test_image(image_output_base, 0.01f, 48));

    std::string msg = "TODO we need a good description here";
    ASCENT_ACTIONS_DUMP(actions, image_output_base, msg);
}

#if 0

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_tire)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtk_disabled(n))
    {
        return;
    }

    //
    // Open an example mesh.
    //
    Node data, verify_info;
    const std::string root_file = 
        conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),
                                       "tire_blueprint.cycle_000000.root");

    conduit::relay::io::blueprint::load_mesh(root_file, data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    ASCENT_INFO("Testing xray_extract on tire example\n");

    const std::string query_output_path = prepare_output_dir();
    const std::string query_output_file = 
        conduit::utils::join_file_path(query_output_path,
                                       "tout_rover_xray_tire_blueprint_query");

    // remove old images before rendering
    remove_test_image(query_output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    // field names are pressure, sb, and temperature
    extracts["e1/params/absorption"] = "pressure";
    // extracts["e1/params/emission"] = "pressure";
    extracts["e1/params/filename"] = query_output_file;
    extracts["e1/params/blueprint"] = "json";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    // TODO: Figure out why is this so slow to run
    ascent.execute(actions);
    // TODO can we ask Ascent for the name of the file it wrote?
    // std::cout << ascent.info().to_yaml() << std::endl;
    ascent.close();

    const std::string full_outfile_name = query_output_file + ".cycle_000000.root";

    Node load_mesh;
    conduit::relay::io::blueprint::load_mesh(full_outfile_name, load_mesh);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(load_mesh, verify_info));

    const std::string image_output_path = prepare_output_dir();
    const std::string image_output_base =
        conduit::utils::join_file_path(image_output_path, "tout_rover_xray_tire");

    render_blueprint_result("intensities", image_output_base, load_mesh);

    // TODO we need the render to make an interesting picture. This will be accomplished
    // by working on the basic mesh output and changing the order of the dimensions.
    // TODO we will need to change the baseline when we are making good renders.
    EXPECT_TRUE(check_test_image(image_output_base, 0.01f, 48));
    
    std::string msg = "TODO we need a good description here";
    ASCENT_ACTIONS_DUMP(actions, image_output_base, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_curv2d)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtk_disabled(n))
    {
        return;
    }

    //
    // Open an example mesh.
    //
    Node data, verify_info;
    const std::string root_file = 
        conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),
                                       "curv2d_blueprint.cycle_000048.root");

    conduit::relay::io::blueprint::load_mesh(root_file, data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    ASCENT_INFO("Testing xray_extract on curv2d example\n");

    const std::string query_output_path = prepare_output_dir();
    const std::string query_output_file = 
        conduit::utils::join_file_path(query_output_path,
                                       "tout_rover_xray_curv2d_blueprint_query");

    // remove old images before rendering
    remove_test_image(query_output_file, 48);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/absorption"] = "d";
    extracts["e1/params/emission"] = "p";
    extracts["e1/params/filename"] = query_output_file;
    extracts["e1/params/blueprint"] = "json";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    // TODO: It seems that rover doesn't support datasets defined using z/r
    ascent.execute(actions);
    // TODO can we ask Ascent for the name of the file it wrote?
    // std::cout << ascent.info().to_yaml() << std::endl;
    ascent.close();

    const std::string full_outfile_name = query_output_file + "_000048.cycle_000048.root";

    Node load_mesh;
    conduit::relay::io::blueprint::load_mesh(full_outfile_name, load_mesh);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(load_mesh, verify_info));

    const std::string image_output_path = prepare_output_dir();
    const std::string image_output_base =
        conduit::utils::join_file_path(image_output_path, "tout_rover_xray_curv2d");

    render_blueprint_result("intensities", image_output_base, load_mesh);

    // TODO we need the render to make an interesting picture. This will be accomplished
    // by working on the basic mesh output and changing the order of the dimensions.
    // TODO we will need to change the baseline when we are making good renders.
    EXPECT_TRUE(check_test_image(image_output_base, 0.01f, 48));
    
    std::string msg = "TODO we need a good description here";
    ASCENT_ACTIONS_DUMP(actions, image_output_base, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_serial_image_params)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtk_disabled(n))
    {
        return;
    }

    //
    // Create an example mesh.
    //

    Node data;
    Node verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
    data.print();
    // get_valid_test_data(data);

    ASCENT_INFO("Testing xray_extract\n");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_rover_xray_params");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "xray";
    // populate some param examples
    extracts["e1/params/absorption"] = "radial";
    extracts["e1/params/precision"] = "single";
    extracts["e1/params/filename"] = output_file;
    extracts["e1/params/image_params/min_value"] = 0.006f;
    extracts["e1/params/image_params/max_value"] = 1.000;
    extracts["e1/params/unit_scalar"] = 0.001f;
    extracts["e1/params/image_params/log_scale"] = "true";

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM 0.0001f
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file));

    std::string msg = "An example of using the xray extract.";
    ASCENT_ACTIONS_DUMP(actions, output_file, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_serial)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtk_disabled(n))
    {
        return;
    }

    //
    // Create an example mesh.
    //
    
    Node data;
    get_valid_test_data(data);

    ASCENT_INFO("Testing xray_extract\n");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path, "tout_rover_xray");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "xray";
    extracts["e1/params/absorption"] = "radial";
    extracts["e1/params/emission"] = "radial";
    extracts["e1/params/filename"] = output_file;

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM 0.0001f
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file));

    std::string msg = "An example of using the xray extract.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//
// Rover Volume tests
//
// Note: Ascent doesn't currently use rover for volume rendering
//

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_volume_min_max)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtk_disabled(n))
    {
        return;
    }

    //
    // Create an example mesh.
    //

    Node data;
    get_valid_test_data(data);
    
    ASCENT_INFO("Testing volume_extract\n");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_rover_volume_min_max");

    // remove old images before rendering
    remove_test_image(output_file);


    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "volume";
    // populate some param examples
    extracts["e1/params/field"] = "radial";
    extracts["e1/params/min_value"] = -1.0;
    extracts["e1/params/emission"] = "radial";
    extracts["e1/params/precision"] = "double";
    extracts["e1/params/filename"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    std::string msg = "An example of using the volume (unstructured grid) extract with "
                      "min and max values.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_volume_serial)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(is_vtk_disabled(n))
    {
        return;
    }

    //
    // Create an example mesh.
    //

    Node data;
    get_valid_test_data(data);

    ASCENT_INFO("Testing volume_extract\n");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_rover_volume");

    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"]  = "volume";
    // populate some param examples
    extracts["e1/params/field"] = "radial";
    extracts["e1/params/filename"] = output_file;

    conduit::Node actions;
    // add the pipeline
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

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
    std::string msg = "An example of using the volume (unstructured grid) extract.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

#endif
