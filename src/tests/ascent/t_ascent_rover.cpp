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

void
execute_ascent(const Node& data,
               const Node& actions)
{
    Ascent ascent;
    Node ascent_opts;
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    // TODO can we ask Ascent for the name of the file it wrote?
    // std::cout << ascent.info().to_yaml() << std::endl;
    ascent.close();
}

void
render_blueprint(const string &field_name,
                 const string &output_path,
                 const Node &data,
                 const double detector_width)
{
    // Define Ascent actions
    // TODO: Remove this once issue #1559 is fixed
    Node pipelines;
    Node &pl = pipelines["pl1"];
    pl["f1/type"] = "clip";
    pl["f1/params/topology"] = "image_topo";
    pl["f1/params/invert"] = "true";
    pl["f1/params/box/min/x"] = 0.0;
    pl["f1/params/box/min/y"] = 0.0;
    pl["f1/params/box/min/z"] = 0.0;
    pl["f1/params/box/max/x"] = detector_width;
    pl["f1/params/box/max/y"] = detector_width;
    pl["f1/params/box/max/z"] = 0.0;

    Node scenes;
    if (field_name.find("spatial") != std::string::npos)
    {
        scenes["s1/renders/r1/camera/azimuth"] = 45.0;
        
        // TODO: Remove this once issue #1559 is fixed
        scenes["s1/plots/p1/pipeline"] = "pl1";
    }

    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = field_name;
    scenes["s1/renders/r1/image_prefix"] = output_path;

    Node actions;
    // TODO: Remove this once issue #1559 is fixed
    Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    // Execute Ascent actions
    execute_ascent(data, actions);
}

void
render_all_fields(const Node &data,
                  const std::string output_path,
                  const int cycle)
{
    // This is here to help identify which ascent execute is throwing an error
    ASCENT_INFO("Executing render_all_fields\n");

    // TODO: Remove this once issue #1559 is fixed
    const Node &xray_data = data["domain_000000/state/xray_data"];
    const double detector_width = xray_data["detector_width"].to_double();
    
    // TODO: Undo this once optical_depth is fixed
    const std::vector<std::string> fields {"intensities", 
                                        //    "optical_depth",
                                           "intensities_spatial",
                                        //    "optical_depth_spatial"
                                          };
    for (auto field : fields)
    {
        std::string full_output_path = output_path + "_" + field;
        render_blueprint(field, full_output_path, data, detector_width);
        EXPECT_TRUE(check_test_image(full_output_path, 0.01f, cycle));
    }
}

void
load_and_verify_local_data(Node &data,
                           const std::string data_path)
{
    Node verify_info;
    conduit::relay::io::blueprint::load_mesh(data_path, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
}

void
load_and_verify_ascent_data(Node &baseline_data,
                            const std::string filename)
{
    Node verify_info;
    const std::string baseline_path = conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR), filename);
    conduit::relay::io::blueprint::load_mesh(baseline_path, baseline_data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(baseline_data, verify_info));
}

void
get_valid_test_data(Node &data)
{
    Node verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                                     EXAMPLE_MESH_SIDE_DIM,
                                                     EXAMPLE_MESH_SIDE_DIM,
                                                     EXAMPLE_MESH_SIDE_DIM,
                                                   data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
}

const bool
is_vtkm_disabled()
{
    Node n;
    ascent::about(n);
    const bool disabled = "disabled" == n["runtimes/ascent/vtkm/status"].as_string();
    if (disabled)
    {
        ASCENT_INFO("Ascent was built without vtkm, skipping test\n");
    }
    return disabled;
}

//
// Rover X-Ray tests
//

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid)
{
    ASCENT_INFO("Testing xray extract on conduit braid example\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid";
    const std::string query_ext_name = "_000100.cycle_000100.root";
    const std::string image_name = "tout_rover_xray_blueprint_braid";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 100;
    remove_test_image(query_path, cycle);

    // Generate and verify test data
    Node test_data;
    get_valid_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";

    Node actions;
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, image_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of an example braid mesh";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_rotated)
{
    ASCENT_INFO("Testing xray extract on conduit braid example (rotated)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid_rotated";
    const std::string query_ext_name = "_000100.cycle_000100.root";
    const std::string image_name = "tout_rover_xray_blueprint_braid_rotated";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 100;
    remove_test_image(query_path, cycle);

    // Generate and verify test data
    Node test_data;
    get_valid_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";
    extracts["e1/params/camera/azimuth"] = 45.0;
    extracts["e1/params/camera/elevation"] = 45.0;

    Node actions;
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, image_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of an example braid mesh (rotated)";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_diff)
{
    ASCENT_INFO("Testing xray extract on conduit braid example (blueprint diff)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid_diff";
    const std::string query_ext_name = "_000100.cycle_000100.root";
    const std::string image_name = "tout_rover_xray_blueprint_braid_diff";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 100;
    remove_test_image(query_path, cycle);

    // Generate and verify test data
    Node test_data;
    get_valid_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";

    Node actions;
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data, diff_info;
    const std::string baseline_filename = "rover_xray_blueprint_braid_baseline.root";
    load_and_verify_ascent_data(baseline_data, baseline_filename);

    // Diff the baseline data with our new output
    const bool has_differences = baseline_data.diff(xray_blueprint_output, diff_info, 0.01, true);
    if (has_differences)
    {
        ASCENT_INFO("Found differences in the braid blueprint diff:\n");
        diff_info.print();
    }
    EXPECT_FALSE(has_differences);

    // Dump info
    std::string msg = "XRay blueprint diff of an example braid mesh";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

#if 0
//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_lowres)
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

    ASCENT_INFO("Testing lowres xray_extract on conduit braid example\n");

    const std::string query_output_path = prepare_output_dir();
    const std::string query_output_file = 
        conduit::utils::join_file_path(query_output_path, "tout_rover_xray_lowres_query");

    // remove old images before rendering
    remove_test_image(query_output_file);

    //
    // Create the actions.
    //

    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = query_output_file;
    extracts["e1/params/rover/blueprint"] = "json";

    // Output resolution
    extracts["e1/params/rover/width"] = 11;
    extracts["e1/params/rover/height"] = 11;

    // Image params
    extracts["e1/params/image_params/min_value"] = 0.006;
    extracts["e1/params/image_params/max_value"] = 1.000;
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
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    const std::string full_outfile_name = query_output_file + "100.cycle_000100.root";

    Node load_mesh, verify_info;
    conduit::relay::io::blueprint::load_mesh(full_outfile_name, load_mesh);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(load_mesh, verify_info));

    const std::string image_output_path = prepare_output_dir();
    const std::string image_output_base =
        conduit::utils::join_file_path(image_output_path, "tout_rover_xray_lowres_blueprint");

    const std::string image_output_image_prefix = image_output_base + "{cycle:d}";

    render_blueprint_result("intensities", image_output_image_prefix, load_mesh);

    EXPECT_TRUE(check_test_image(image_output_base, 0.01f));

    std::string msg = "Render a lowres XRay diagnostic image of an example braid mesh";
    ASCENT_ACTIONS_DUMP(actions, image_output_base, msg);
}
#endif

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_curv3d)
{
    ASCENT_INFO("Testing xray extract on curv3d data\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_curv3d";
    const std::string query_ext_name = "_000048.cycle_000048.root";
    const std::string image_name = "tout_rover_xray_blueprint_curv3d";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 48;
    remove_test_image(query_path, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "d";
    extracts["e1/params/rover/emission"] = "p";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, image_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the curv3d dataset";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_curv3d_rotated)
{
    ASCENT_INFO("Testing xray extract on curv3d data (rotated)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_curv3d_rotated";
    const std::string query_ext_name = "_000048.cycle_000048.root";
    const std::string image_name = "tout_rover_xray_blueprint_curv3d_rotated";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 48;
    remove_test_image(query_path, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "d";
    extracts["e1/params/rover/emission"] = "p";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";
    extracts["e1/params/camera/azimuth"] = 45.0;
    extracts["e1/params/camera/elevation"] = 45.0;

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, image_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the curv3d dataset (rotated)";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_curv3d_diff)
{
    ASCENT_INFO("Testing xray extract on curv3d data (blueprint diff)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_curv3d";
    const std::string query_ext_name = "_000048.cycle_000048.root";
    const std::string image_name = "tout_rover_xray_blueprint_curv3d";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 48;
    remove_test_image(query_path, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "d";
    extracts["e1/params/rover/emission"] = "p";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data, diff_info;
    const std::string baseline_filename = "rover_xray_blueprint_curv3d_baseline.root";
    load_and_verify_ascent_data(baseline_data, baseline_filename);

    // Diff the baseline data with our new output
    const bool has_differences = baseline_data.diff(xray_blueprint_output, diff_info, 0.01, true);
    if (has_differences)
    {
        ASCENT_INFO("Found differences in the curv3d blueprint diff:\n");
        diff_info.print();
    }
    EXPECT_FALSE(has_differences);

    // Dump info
    std::string msg = "XRay blueprint diff of the curv3d dataset";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_curv3d_camera_params)
{
    ASCENT_INFO("Testing xray extract on curv3d data (all camera params)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_curv3d_camera_params";
    const std::string query_ext_name = "_000048.cycle_000048.root";
    const std::string image_name = "tout_rover_xray_blueprint_curv3d_camera_params";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 48;
    remove_test_image(query_path, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "d";
    extracts["e1/params/rover/emission"] = "p";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";

    // These errors all originate from within rover
    // TODO: Setting anything for position (e.g. 0,0,0) throws a vector range error
    // TODO: Setting (0,0,0) for up throws a vector range error
    // TODO: If xpan = 1 and ypan = 2, throws a vector range error

    // Change all of the default camera parameters
    double vec3[3] = {1.0, 1.0, 1.0};
    extracts["e1/params/camera/look_at"].set_float64_ptr(vec3, 3);
    // extracts["e1/params/camera/position"].set_float64_ptr(vec3, 3);
    extracts["e1/params/camera/up"].set_float64_ptr(vec3, 3);
    extracts["e1/params/camera/fov"] = 60.0;
    // extracts["e1/params/camera/xpan"] = -0.1;
    // extracts["e1/params/camera/ypan"] = 0.1;
    extracts["e1/params/camera/zoom"] = 1.5;
    extracts["e1/params/camera/near_plane"] = 2.0;
    extracts["e1/params/camera/far_plane"] = 50.0;

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, image_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the curv3d dataset (all camera params)";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_multi_curv3d)
{
    ASCENT_INFO("Testing xray extract on multi_curv3d data\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_multi_curv3d";
    const std::string query_ext_name = "_000048.cycle_000048.root";
    const std::string image_name = "tout_rover_xray_blueprint_multi_curv3d";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 48;
    remove_test_image(query_path, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "multi_curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "d";
    extracts["e1/params/rover/emission"] = "p";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, image_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the multi_curv3d dataset";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_multi_curv3d_rotated)
{
    ASCENT_INFO("Testing xray extract on multi_curv3d data (rotated)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_multi_curv3d_rotated";
    const std::string query_ext_name = "_000048.cycle_000048.root";
    const std::string image_name = "tout_rover_xray_blueprint_multi_curv3d_rotated";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                 query_name);
    const std::string output_data_path = query_path + query_ext_name;
    const std::string image_path = conduit::utils::join_file_path(output_path,
                                                                 image_name);

    // Remove old test image
    const int cycle = 48;
    remove_test_image(query_path, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "multi_curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    conduit::Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "d";
    extracts["e1/params/rover/emission"] = "p";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/blueprint"] = "yaml";
    extracts["e1/params/camera/azimuth"] = 45.0;
    extracts["e1/params/camera/elevation"] = 45.0;

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, image_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the multi_curv3d dataset (rotated)";
    ASCENT_ACTIONS_DUMP(actions, image_path, msg);
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
    extracts["e1/params/rover/absorption"] = "pressure";
    // extracts["e1/params/rover/emission"] = "pressure";
    extracts["e1/params/rover/filename"] = query_output_file;
    extracts["e1/params/rover/blueprint"] = "json";

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
    EXPECT_TRUE(check_test_image(image_output_base, 0.01f, 48));
    
    std::string msg = "Render an XRay diagnostic image of the tire mesh";
    ASCENT_ACTIONS_DUMP_CYCLE(actions, image_output_base, msg, 48);
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
    extracts["e1/params/rover/absorption"] = "d";
    extracts["e1/params/rover/emission"] = "p";
    extracts["e1/params/rover/filename"] = query_output_file;
    extracts["e1/params/rover/blueprint"] = "json";

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
    EXPECT_TRUE(check_test_image(image_output_base, 0.01f, 48));
    
    std::string msg = "Render an XRay diagnostic image of the curv2d mesh";
    ASCENT_ACTIONS_DUMP_CYCLE(actions, image_output_base, msg, 48);
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
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/precision"] = "single";
    extracts["e1/params/rover/filename"] = output_file;
    extracts["e1/params/rover/unit_scalar"] = 0.001f;
    extracts["e1/params/image_params/min_value"] = 0.006f;
    extracts["e1/params/image_params/max_value"] = 1.000;
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
    ascent_opts["exceptions"] = "forward";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // check that we created an image
    // NOTE: RELAXED TOLERANCE TO FROM 0.0001f
    //       to mitigate differences between platforms
    EXPECT_TRUE(check_test_image(output_file, 0.01f));

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
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = output_file;

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
    EXPECT_TRUE(check_test_image(output_file, 0.01f));

    std::string msg = "An example of using the xray extract.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
#endif

#if 0 // removing volume renderer
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
    EXPECT_TRUE(check_test_image(output_file, 0.01f));
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
    EXPECT_TRUE(check_test_image(output_file, 0.01f));
    std::string msg = "An example of using the volume (unstructured grid) extract.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
#endif
