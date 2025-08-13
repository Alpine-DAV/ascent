//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_mpi_rover.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <math.h>
#include <mpi.h>

#include <conduit_blueprint_mpi.hpp>

#include "rover_test_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

// MPI variables that get used everywhere
int par_rank = 0;
int par_size = 1;

//
// MPI Rover X-Ray tests
//

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_mpi_blueprint_braid_uniform_multi_domain)
{
    // Set up MPI
    MPI_Comm_rank(COMM, &par_rank);
    MPI_Comm_size(COMM, &par_size);
    const bool is_root = (0 == par_rank);

    if (is_root)
    {
        ASCENT_INFO("Testing xray extract using MPI on a conduit braid_uniform_multi_domain example mesh\n");
    }

    if (is_vtkm_disabled(is_root))
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_mpi_blueprint_braid_uniform_multi_domain";
    const std::string query_suffix = "_000000.cycle_000000.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    remove_rover_test_data(query_path, query_suffix);

    // Generate test data
    Node test_data;
    get_mpi_braid_multi_domain_test_data(test_data, par_rank, par_size);

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
    Node xray_blueprint_output;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Rover's output is only single-domain, so we only use rank 0 to verify the output
    if (is_root)
    {
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
    }

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path);

    // We only want to dump info if we are rank 0
    if (is_root)
    {
        // Dump info
        const std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh";
        ASCENT_ACTIONS_DUMP(actions, query_path, msg);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_mpi_blueprint_braid_uniform_multi_domain_rotated)
{
    // Set up MPI
    MPI_Comm_rank(COMM, &par_rank);
    MPI_Comm_size(COMM, &par_size);
    const bool is_root = (0 == par_rank);

    if (is_root)
    {
        ASCENT_INFO("Testing xray extract using MPI on a conduit braid_uniform_multi_domain example mesh (rotated)\n");
    }

    if (is_vtkm_disabled(is_root))
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_mpi_blueprint_braid_uniform_multi_domain_rotated";
    const std::string query_suffix = "_000000.cycle_000000.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;
    
    // Remove old test data
    remove_rover_test_data(query_path, query_suffix);

    // Generate test data
    Node test_data;
    get_mpi_braid_multi_domain_test_data(test_data, par_rank, par_size);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "radial");
    extracts["e1/params/rover/background_intensity"] = 12.34;
    extracts["e1/params/rover/enable_rays_mesh"] = "true";
    const double azimuth = 60.0;
    add_camera_rotation(extracts, azimuth);

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);
    
    // Load and verify output mesh
    Node xray_blueprint_output;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Rover's output is only single-domain, so we only use rank 0 to verify the output
    if (is_root)
    {
        // Load and verify baseline data
        Node baseline_data;
        get_default_baseline(baseline_data, extracts["e1/params"]);
    
        // Manually override the remaining fields with expected values
        baseline_data["time"] = 3.1414999961853;
        baseline_data["xray_view/position"] = {40.0, 34.6410140991211, 17.3205070495605};
        baseline_data["xray_view/look_at"] = {10.0, 0.0, 0.0};
        baseline_data["xray_view/near_plane"] = 4.89897966384888;
        baseline_data["xray_view/far_plane"] = 489.89794921875;
        baseline_data["xray_data/detector_width"] = 5.65685440236809;
        baseline_data["xray_data/detector_height"] = 5.65685440236809;
        baseline_data["xray_data/intensity_max"] = 173.205078125;
        baseline_data["xray_data/intensity_min"] = 12.3400001525879;
        baseline_data["xray_data/optical_depth_max"] = 3120.77880859375;

        // Diff the baseline data with our new output
        const Node &state_output = xray_blueprint_output["domain_000000/state"];
        check_blueprint_diff(baseline_data, state_output);
    }

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path);

    // We only want to dump info if we are rank 0
    if (is_root)
    {
        // Dump info
        const std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh (rotated)";
        ASCENT_ACTIONS_DUMP(actions, query_path, msg);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_mpi_blueprint_braid_uniform_multi_domain_absorption_only)
{
    // Set up MPI
    MPI_Comm_rank(COMM, &par_rank);
    MPI_Comm_size(COMM, &par_size);
    const bool is_root = (0 == par_rank);

    if (is_root)
    {
        ASCENT_INFO("Testing xray extract using MPI on a conduit braid_uniform_multi_domain example mesh (absorption only)\n");
    }

    if (is_vtkm_disabled(is_root))
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_mpi_blueprint_braid_uniform_multi_domain_absorption_only";
    const std::string query_suffix = "_000000.cycle_000000.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;

    // Remove old test data
    remove_rover_test_data(query_path, query_suffix);

    // Generate test data
    Node test_data;
    get_mpi_braid_multi_domain_test_data(test_data, par_rank, par_size);

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "", "json");
    extracts["e1/params/rover/precision"] = "double";
    extracts["e1/params/rover/unit_scalar"] = 1.234;

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);
    
    // Load and verify output mesh
    Node xray_blueprint_output;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Rover's output is only single-domain, so we only use rank 0 to verify the output
    if (is_root)
    {
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
        baseline_data["xray_data/optical_depth_max"] = 3683.56120526528;

        // Diff the baseline data with our new output
        const Node &state_output = xray_blueprint_output["domain_000000/state"];
        check_blueprint_diff(baseline_data, state_output);
    }

    // Render and verify each field
    const int cycle = 0;
    const bool render_intensities = false;
    render_fields(xray_blueprint_output, query_path, cycle, render_intensities);

    // We only want to dump info if we are rank 0
    if (is_root)
    {
        // Dump info
        const std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh (absorption only)";
        ASCENT_ACTIONS_DUMP(actions, query_path, msg);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_mpi_blueprint_braid_uniform_single_domain_multiple_ranks)
{
    // Set up MPI
    MPI_Comm_rank(COMM, &par_rank);
    MPI_Comm_size(COMM, &par_size);
    const bool is_root = (0 == par_rank);

    if (is_root)
    {
        ASCENT_INFO("Testing xray extract using MPI on a conduit braid_uniform_single_domain example mesh\n");
    }

    if (is_vtkm_disabled(is_root))
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_mpi_blueprint_braid_uniform_single_domain";
    const std::string query_suffix = "_000000.cycle_000000.root";

    // Set up paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = utils::join_file_path(output_path, query_name);
    const std::string output_data_path = query_path + query_suffix;
    
    // Remove old test data
    remove_rover_test_data(query_path, query_suffix);

    // Generate test data
    Node test_data;

    // We only want rank 0 to have data so that we test the case in which
    // multiple ranks are used but not all of them have data
    if (is_root)
    {
        get_braid_multi_domain_test_data(test_data, 1);
    }

    // Define Ascent actions
    Node extracts;
    get_common_extract_params(extracts, query_path, "radial", "radial");

    Node actions;
    get_default_action_params(actions, extracts);

    // Execute Ascent actions
    execute_ascent(test_data, actions);
    
    // Load and verify output mesh
    Node xray_blueprint_output;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    // Rover's output is only single-domain, so we only use rank 0 to verify the output
    if (is_root)
    {
        // Load and verify baseline data
        Node baseline_data;
        get_default_baseline(baseline_data, extracts["e1/params"]);

        // Manually override the remaining fields with expected values
        baseline_data["time"] = 3.1414999961853;
        baseline_data["xray_view/position"] = {0.0, 0.0, 34.6410179138184};
        baseline_data["xray_view/look_at"] = {0.0, 0.0, 0.0};
        baseline_data["xray_view/near_plane"] = 3.46410179138184;
        baseline_data["xray_view/far_plane"] = 346.410186767578;
        baseline_data["xray_data/detector_width"] = 4.00000016604152;
        baseline_data["xray_data/detector_height"] = 4.00000016604152;
        baseline_data["xray_data/intensity_max"] = 173.205078125;
        baseline_data["xray_data/optical_depth_max"] = 2811.4736328125;
    
        // Diff the baseline data with our new output
        const Node &state_output = xray_blueprint_output["domain_000000/state"];
        check_blueprint_diff(baseline_data, state_output);
    }

    // Render and verify each field
    render_fields(xray_blueprint_output, query_path);

    // We only want to dump info if we are rank 0
    if (is_root)
    {
        // Dump info
        const std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh";
        ASCENT_ACTIONS_DUMP(actions, query_path, msg);
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
