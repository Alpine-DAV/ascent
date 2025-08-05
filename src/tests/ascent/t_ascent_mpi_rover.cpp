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

#include "t_rover_test_utils.hpp"

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

    if (0 == par_rank)
    {
        ASCENT_INFO("Testing xray extract using MPI on a conduit braid_uniform_multi_domain example mesh\n");
    }

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_mpi_blueprint_braid_uniform_multi_domain";
    const std::string query_ext_name = "_000000.cycle_000000.root";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;
    
    // Generate test data
    Node test_data;
    blueprint::mpi::mesh::examples::braid_uniform_multi_domain(test_data, COMM);

    // Verify test data
    Node verify_test_data;
    EXPECT_TRUE(conduit::blueprint::mpi::mesh::verify(test_data, verify_test_data, COMM));

    // Define Ascent actions
    Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/output_type"] = "json";
    extracts["e1/params/rover/precision"] = "double";
    extracts["e1/params/rover/unit_scalar"] = 1.234f;

    Node actions;
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);
    
    // Load and verify output mesh
    Node xray_blueprint_output;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    if (0 == par_rank)
    {
        Node &state_output = xray_blueprint_output["domain_000000/state"];
    
        // Load and verify baseline data
        const std::string yaml = R"yaml(
            time: 3.1414999961853
            cycle: 0
            xray_view: 
                position: [10.0, 0.0, 48.9897956848145]
                zoom: 1.0
                look_at: [10.0, 0.0, 0.0]
                up: [0.0, 1.0, 0.0]
                fov: 60.0
                xpan: 0.0
                ypan: 0.0
                near_plane: 4.89897966384888
                far_plane: 489.89794921875
            xray_query: 
                background_intensity: 0.0
                divide_emis_by_absorb: "false"
                enable_rays_mesh: "false"
                height: 200
                precision: "double"
                width: 200
                unit_scalar: 1.23399996757507
                absorption: "radial"
                emission: "radial"
                filename: ""
                output_type: "json"
            xray_data: 
                detector_width: 5.65685440236809
                detector_height: 5.65685440236809
                intensity_max: 213.735064037837
                intensity_min: 0.0
                optical_depth_max: 3683.56120526528
                optical_depth_min: 0.0
                image_topo_order_of_domain_variables: "xyz"
            domain_id: 0
        )yaml";
    
        Node baseline_data;
        baseline_data.parse(yaml);
        baseline_data["xray_query/filename"] = query_path;
    
        // Diff the baseline data with our new output
        check_blueprint_diff(baseline_data, state_output);
    }

    // Render and verify each field
    const int cycle = 0;
    render_fields(xray_blueprint_output, query_path, cycle);

    if (0 == par_rank)
    {
        // Dump info
        std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh";
        ASCENT_ACTIONS_DUMP(actions, query_path, msg);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_mpi_blueprint_braid_uniform_multi_domain_rotated)
{
    // Set up MPI
    MPI_Comm_rank(COMM, &par_rank);
    MPI_Comm_size(COMM, &par_size);

    if (0 == par_rank)
    {
        ASCENT_INFO("Testing xray extract using MPI on a conduit braid_uniform_multi_domain example mesh (rotated)\n");
    }

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_mpi_blueprint_braid_uniform_multi_domain_rotated";
    const std::string query_ext_name = "_000000.cycle_000000.root";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;
    
    // Generate test data
    Node test_data;
    blueprint::mpi::mesh::examples::braid_uniform_multi_domain(test_data, COMM);

    // Verify test data
    Node verify_test_data;
    EXPECT_TRUE(conduit::blueprint::mpi::mesh::verify(test_data, verify_test_data, COMM));

    // Define Ascent actions
    Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/output_type"] = "yaml";
    extracts["e1/params/rover/background_intensity"] = 12.34f;
    extracts["e1/params/rover/enable_rays_mesh"] = "true";
    extracts["e1/params/camera/azimuth"] = 60.0;
    extracts["e1/params/camera/elevation"] = 45.0;

    Node actions;
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);
    
    // Load and verify output mesh
    Node xray_blueprint_output;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    if (0 == par_rank)
    {
        Node &state_output = xray_blueprint_output["domain_000000/state"];
    
        // Load and verify baseline data
        const std::string yaml = R"yaml(
            time: 3.1414999961853
            cycle: 0
            xray_view: 
                position: [40.0, 34.6410140991211, 17.3205070495605]
                zoom: 1.0
                look_at: [10.0, 0.0, 0.0]
                up: [0.0, 1.0, 0.0]
                fov: 60.0
                xpan: 0.0
                ypan: 0.0
                near_plane: 4.89897966384888
                far_plane: 489.89794921875
            xray_query: 
                background_intensity: 12.3400001525879
                divide_emis_by_absorb: "false"
                enable_rays_mesh: "true"
                height: 200
                precision: "single"
                width: 200
                unit_scalar: 1.0
                absorption: "radial"
                emission: "radial"
                filename: ""
                output_type: "yaml"
                camera: 
                    azimuth: 60.0
                    elevation: 45.0
            xray_data: 
                detector_width: 5.65685440236809
                detector_height: 5.65685440236809
                intensity_max: 173.205078125
                intensity_min: 12.3400001525879
                optical_depth_max: 3120.77880859375
                optical_depth_min: 0.0
                image_topo_order_of_domain_variables: "xyz"
            domain_id: 0
        )yaml";
    
        Node baseline_data;
        baseline_data.parse(yaml);
        baseline_data["xray_query/filename"] = query_path;
    
        // Diff the baseline data with our new output
        check_blueprint_diff(baseline_data, state_output);
    }

    // Render and verify each field
    const int cycle = 0;
    render_fields(xray_blueprint_output, query_path, cycle);

    if (0 == par_rank)
    {
        // Dump info
        std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh (rotated)";
        ASCENT_ACTIONS_DUMP(actions, query_path, msg);
    }
}

//-----------------------------------------------------------------------------
TEST(ascent_rover, test_xray_mpi_blueprint_braid_uniform_single_domain_multiple_ranks)
{
    // Set up MPI
    MPI_Comm_rank(COMM, &par_rank);
    MPI_Comm_size(COMM, &par_size);

    if (0 == par_rank)
    {
        ASCENT_INFO("Testing xray extract using MPI on a conduit braid_uniform_single_domain example mesh (\n");
    }

    if (is_vtkm_disabled())
    {
        return; // Returning early is equivalent to passing the test
    }

    // Test names
    const std::string query_name = "tout_rover_xray_mpi_blueprint_braid_uniform_single_domain";
    const std::string query_ext_name = "_000000.cycle_000000.root";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;
    
    // Generate test data
    Node test_data;
    blueprint::mpi::mesh::examples::braid_uniform_multi_domain(test_data, COMM);

    // Verify test data
    Node verify_test_data;
    EXPECT_TRUE(conduit::blueprint::mpi::mesh::verify(test_data, verify_test_data, COMM));

    // This ensures that only rank 0 publishes data, so that we
    // test the case in which at least one rank has nothing to do
    if (0 != par_rank)
    {
        test_data.reset();
    }

    // Define Ascent actions
    Node extracts;
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/absorption"] = "radial";
    extracts["e1/params/rover/emission"] = "radial";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/output_type"] = "yaml";

    Node actions;
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);
    
    // Load and verify output mesh
    Node xray_blueprint_output;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);

    if (0 == par_rank)
    {
        Node &state_output = xray_blueprint_output["domain_000000/state"];
    
        // Load and verify baseline data
        const std::string yaml = R"yaml(
            time: 3.1414999961853
            cycle: 0
            xray_view: 
                position: [0.0, 0.0, 34.6410179138184]
                zoom: 1.0
                look_at: [0.0, 0.0, 0.0]
                up: [0.0, 1.0, 0.0]
                fov: 60.0
                xpan: 0.0
                ypan: 0.0
                near_plane: 3.46410179138184
                far_plane: 346.410186767578
            xray_query: 
                background_intensity: 0.0
                divide_emis_by_absorb: "false"
                enable_rays_mesh: "false"
                height: 200
                precision: "single"
                width: 200
                unit_scalar: 1.0
                absorption: "radial"
                emission: "radial"
                filename: ""
                output_type: "yaml"
            xray_data: 
                detector_width: 4.00000016604152
                detector_height: 4.00000016604152
                intensity_max: 173.205078125
                intensity_min: 0.0
                optical_depth_max: 2811.4736328125
                optical_depth_min: 0.0
                image_topo_order_of_domain_variables: "xyz"
            domain_id: 0
        )yaml";
    
        Node baseline_data;
        baseline_data.parse(yaml);
        baseline_data["xray_query/filename"] = query_path;
    
        // Diff the baseline data with our new output
        check_blueprint_diff(baseline_data, state_output);
    }

    // Render and verify each field
    const int cycle = 0;
    render_fields(xray_blueprint_output, query_path, cycle);

    if (0 == par_rank)
    {
        // Dump info
        std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh";
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
