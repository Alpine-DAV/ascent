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

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "rover_test_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

//
// Rover X-Ray tests
//

// TODO: Create an absorption-only multi-group test
// TODO: Create an imaging planes and rays mesh test

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid)
{
    ASCENT_INFO("Testing xray extract on conduit braid example mesh\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid";
    const std::string query_suffix = "_000100.cycle_000100.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 100;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Generate and verify test data
    Node test_data;
    get_braid_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "radial");
    extracts["e1/params/rover/precision"] = "double";

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 3.1414999961853;
    baseline_data["xray_view/position"] = {0.0, 0.0, 34.6410179138184};
    baseline_data["xray_view/near_plane"] = 3.46410179138184;
    baseline_data["xray_view/far_plane"] = 346.410186767578;
    baseline_data["xray_data/detector_width"] = 4.00000016604152;
    baseline_data["xray_data/detector_height"] = 4.00000016604152;
    baseline_data["xray_data/intensity_max"] = 173.205078125;
    baseline_data["xray_data/optical_depth_max"] = 2698.02783203125;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of an example braid mesh";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_rotated)
{
    ASCENT_INFO("Testing xray extract on conduit braid example mesh (rotated)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid_rotated";
    const std::string query_suffix = "_000100.cycle_000100.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 100;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Generate and verify test data
    Node test_data;
    get_braid_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "radial", "json");
    extracts["e1/params/rover/background_intensity"] = 100.0;
    add_camera_rotation(extracts);

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 3.1414999961853;
    baseline_data["xray_view/position"] = {17.3205070495605, 24.49489402771, 17.3205070495605};
    baseline_data["xray_view/near_plane"] = 3.46410179138184;
    baseline_data["xray_view/far_plane"] = 346.410186767578;
    baseline_data["xray_data/detector_width"] = 4.00000016604152;
    baseline_data["xray_data/detector_height"] = 4.00000016604152;
    baseline_data["xray_data/intensity_max"] = 173.205078125;
    baseline_data["xray_data/intensity_min"] = 100.0;
    baseline_data["xray_data/optical_depth_max"] = 2475.25146484375;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of an example braid mesh (rotated)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_absorption_only)
{
    ASCENT_INFO("Testing xray extract on conduit braid example mesh (absorption only)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid_absorption_only";
    const std::string query_suffix = "_000100.cycle_000100.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 100;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Generate and verify test data
    Node test_data;
    get_braid_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "");
    extracts["e1/params/rover/precision"] = "double";

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 3.1414999961853;
    baseline_data["xray_view/position"] = {0.0, 0.0, 34.6410179138184};
    baseline_data["xray_view/near_plane"] = 3.46410179138184;
    baseline_data["xray_view/far_plane"] = 346.410186767578;
    baseline_data["xray_data/detector_width"] = 4.00000016604152;
    baseline_data["xray_data/detector_height"] = 4.00000016604152;
    baseline_data["xray_data/optical_depth_max"] = 2698.02783203125;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    const bool render_intensities = false;
    render_fields(xray_blueprint_output, query_path, cycle, render_intensities);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of an example braid mesh (absorption only, rotated)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_absorption_only_rotated)
{
    ASCENT_INFO("Testing xray extract on conduit braid example mesh (absorption only, rotated)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid_absorption_only_rotated";
    const std::string query_suffix = "_000100.cycle_000100.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 100;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Generate and verify test data
    Node test_data;
    get_braid_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "");
    extracts["e1/params/rover/precision"] = "double";
    add_camera_rotation(extracts);

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 3.1414999961853;
    baseline_data["xray_view/position"] = {17.3205070495605, 24.49489402771, 17.3205070495605};
    baseline_data["xray_view/near_plane"] = 3.46410179138184;
    baseline_data["xray_view/far_plane"] = 346.410186767578;
    baseline_data["xray_data/detector_width"] = 4.00000016604152;
    baseline_data["xray_data/detector_height"] = 4.00000016604152;
    baseline_data["xray_data/optical_depth_max"] = 2475.25178205037;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    const bool render_intensities = false;
    render_fields(xray_blueprint_output, query_path, cycle, render_intensities);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of an example braid mesh (absorption only, rotated)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_uniform_multi_domain)
{
    ASCENT_INFO("Testing xray extract on a conduit braid_uniform_multi_domain example mesh\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid_uniform_multi_domain";
    const std::string query_suffix = "_000000.cycle_000000.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    remove_rover_test_data(query_path, query_suffix);

    // Generate and verify test data
    Node test_data;
    get_braid_multi_domain_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "radial", "json");
    extracts["e1/params/rover/precision"] = "double";
    extracts["e1/params/rover/unit_scalar"] = 1.234;

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"]);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 3.1414999961853;
    baseline_data["xray_view/position"] = {10.0, 0.0, 48.9897956848145};
    baseline_data["xray_view/look_at"] = {10.0, 0.0, 0.0};
    baseline_data["xray_view/near_plane"] = 4.89897966384888;
    baseline_data["xray_view/far_plane"] = 489.89794921875;
    baseline_data["xray_data/detector_width"] = 5.65685440236809;
    baseline_data["xray_data/detector_height"] = 5.65685440236809;
    baseline_data["xray_data/intensity_max"] = 213.735064037837;
    baseline_data["xray_data/optical_depth_max"] = 3683.56120526528;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path);

    // Dump info
    const std::string msg = "Rendered xray diagnostic images of an example braid_uniform_multi_domain mesh";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_braid_uniform_multi_domain_rotated)
{
    ASCENT_INFO("Testing xray extract on a conduit braid_uniform_multi_domain example mesh (rotated)\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_braid_uniform_multi_domain_rotated";
    const std::string query_suffix = "_000000.cycle_000000.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    remove_rover_test_data(query_path, query_suffix);

    // Generate and verify test data
    Node test_data;
    get_braid_multi_domain_test_data(test_data);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "radial");
    extracts["e1/params/rover/background_intensity"] = 12.34f;
    extracts["e1/params/rover/enable_rays_mesh"] = "true";
    const double azimuth = 60.0;
    add_camera_rotation(extracts, azimuth);

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"]);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 3.1414999961853;
    baseline_data["xray_view/position"] = {40.0, 34.6410140991211, 17.3205070495605};
    baseline_data["xray_view/look_at"] = {10.0, 0.0, 0.0};
    baseline_data["xray_view/near_plane"] = 4.89897966384888;
    baseline_data["xray_view/far_plane"] = 489.89794921875;
    baseline_data["xray_query/background_intensity"] = 12.3400001525879;
    baseline_data["xray_data/detector_width"] = 5.65685440236809;
    baseline_data["xray_data/detector_height"] = 5.65685440236809;
    baseline_data["xray_data/intensity_max"] = 173.205078125;
    baseline_data["xray_data/intensity_min"] = 12.3400001525879;
    baseline_data["xray_data/optical_depth_max"] = 3120.77880859375;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path);

    // Dump info
    const std::string msg = "Rendered xray diagnostic images of an example braid_uniform_multi_domain mesh (rotated)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
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
    extracts["e1/params/rover/output_type"] = "yaml";

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
    const std::string query_suffix = "_000048.cycle_000048.root";
    
    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 48;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "d", "p");
    extracts["e1/params/rover/precision"] = "double";

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 4.80000019073486;
    baseline_data["xray_view/position"] = {0.0, 2.5, 47.0156211853027};
    baseline_data["xray_view/look_at"] = {0.0, 2.5, 15.0};
    baseline_data["xray_view/near_plane"] = 3.20156216621399;
    baseline_data["xray_view/far_plane"] = 320.156219482422;
    baseline_data["xray_data/detector_width"] = 3.69684552235394;
    baseline_data["xray_data/detector_height"] = 3.69684552235394;
    baseline_data["xray_data/intensity_max"] = 0.491446942090988;
    baseline_data["xray_data/optical_depth_max"] = 125.497886657715;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of the curv3d dataset";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
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
    const std::string query_suffix = "_000048.cycle_000048.root";
    
    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 48;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "d", "p", "json");
    add_camera_rotation(extracts);

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 4.80000019073486;
    baseline_data["xray_view/position"] = {16.0078086853027, 25.1384601593018, 31.0078086853027};
    baseline_data["xray_view/look_at"] = {0.0, 2.5, 15.0};
    baseline_data["xray_view/near_plane"] = 3.20156216621399;
    baseline_data["xray_view/far_plane"] = 320.156219482422;
    baseline_data["xray_data/detector_width"] = 3.69684552235394;
    baseline_data["xray_data/detector_height"] = 3.69684552235394;
    baseline_data["xray_data/intensity_max"] = 0.478813946247101;
    baseline_data["xray_data/optical_depth_max"] = 37.0659294128418;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of the curv3d dataset (rotated)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
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
    const std::string query_suffix = "_000048.cycle_000048.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 48;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    Node extracts;
    // TODO: Investigate why using "hdf5" here fails the diff test. Seems to be a Conduit issue?
    get_common_extract_params(extracts, query_path, "d", "p");
    
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

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 4.80000019073486;
    baseline_data["xray_view/position"] = {0.0, 2.5, 47.0156211853027};
    baseline_data["xray_view/zoom"] = 1.5;
    baseline_data["xray_view/look_at"] = {1.0, 1.0, 1.0};
    baseline_data["xray_view/up"] = {0.577350258827209, 0.577350258827209, 0.577350258827209};
    baseline_data["xray_view/near_plane"] = 2.0;
    baseline_data["xray_view/far_plane"] = 50.0;
    baseline_data["xray_data/detector_width"] = 1.53960070341771;
    baseline_data["xray_data/detector_height"] = 1.53960070341771;
    baseline_data["xray_data/intensity_max"] = 0.491446971893311;
    baseline_data["xray_data/optical_depth_max"] = 126.027252197266;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of the curv3d dataset (all camera params)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_blueprint_multiple_groups)
{
    ASCENT_INFO("Testing xray extract on multi-group curv3d example mesh\n");

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_blueprint_multiple_groups";
    const std::string query_suffix = "_000048.cycle_000048.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 48;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Generate and verify test data
    Node test_data;
    get_multi_group_curv3d_data(test_data);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "d_multi", "p_multi");

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 4.80000019073486;
    baseline_data["xray_view/position"] = {0.0, 2.49999904632568, 47.0156211853027};
    baseline_data["xray_view/look_at"] = {0.0, 2.49999904632568, 15.0};
    baseline_data["xray_view/near_plane"] = 3.20156216621399;
    baseline_data["xray_view/far_plane"] = 320.156219482422;
    baseline_data["xray_data/detector_width"] = 3.69684552235394;
    baseline_data["xray_data/detector_height"] = 3.69684552235394;
    baseline_data["xray_data/intensity_max"] = 2.94868206977844;
    baseline_data["xray_data/optical_depth_max"] = 752.98779296875;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field for multi-group data
    render_multi_group_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of an example multi-group curv3d mesh";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
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
    const std::string query_suffix = "_000048.cycle_000048.root";
    
    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 48;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "multi_curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "d", "p");
    // TODO: Investigate why using double precision with this
    // dataset has an artifact in the intensity output
    // extracts["e1/params/rover/precision"] = "double";
    extracts["e1/params/rover/divide_emis_by_absorb"] = "true";

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 4.80000019073486;
    baseline_data["xray_view/position"] = {0.0, 2.49999904632568, 47.0156211853027};
    baseline_data["xray_view/look_at"] = {0.0, 2.49999904632568, 15.0};
    baseline_data["xray_view/near_plane"] = 3.20156216621399;
    baseline_data["xray_view/far_plane"] = 320.156219482422;
    baseline_data["xray_data/detector_width"] = 3.69684552235394;
    baseline_data["xray_data/detector_height"] = 3.69684552235394;
    baseline_data["xray_data/intensity_max"] = 0.241532012820244;
    baseline_data["xray_data/optical_depth_max"] = 125.49796295166;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of the multi_curv3d dataset";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
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
    const std::string query_suffix = "_000048.cycle_000048.root";
    
    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    const int cycle = 48;
    remove_rover_test_data(query_path, query_suffix, cycle);

    // Load and verify test data
    Node test_data;
    const std::string filename = "multi_curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(test_data, filename);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "d", "p", "json");
    add_camera_rotation(extracts);

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Load and verify baseline data
    Node baseline_data;
    get_default_baseline(baseline_data, extracts["e1/params"], cycle);

    // Manually override the remaining fields with expected values
    baseline_data["time"] = 4.80000019073486;
    baseline_data["xray_view/position"] = {16.0078086853027, 25.1384582519531, 31.0078086853027};
    baseline_data["xray_view/look_at"] = {0.0, 2.49999904632568, 15.0};
    baseline_data["xray_view/near_plane"] = 3.20156216621399;
    baseline_data["xray_view/far_plane"] = 320.156219482422;
    baseline_data["xray_data/detector_width"] = 3.69684552235394;
    baseline_data["xray_data/detector_height"] = 3.69684552235394;
    baseline_data["xray_data/intensity_max"] = 0.478814035654068;
    baseline_data["xray_data/optical_depth_max"] = 37.0657119750977;

    // Diff the baseline data with our new output
    const Node &state_output = xray_blueprint_output["domain_000000/state"];
    check_blueprint_diff(baseline_data, state_output);
    
    // Render and verify each field
    render_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    const std::string msg = "Rendered XRay diagnostic images of the multi_curv3d dataset (rotated)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
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
    extracts["e1/params/rover/output_type"] = "yaml";

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
    extracts["e1/params/rover/output_type"] = "yaml";

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
