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
#include <mpi.h>

#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mpi.hpp>
#include <conduit_relay.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>

#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;

constexpr index_t EXAMPLE_MESH_SIDE_DIM = 20;

// MPI variables that get used everywhere
const MPI_Comm COMM = MPI_COMM_WORLD;
int par_rank = 0;
int par_size = 1;

//
// Utilities
//

void
execute_ascent(const Node &data,
               const Node &actions)
{
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["runtime"] = "ascent";
    ascent_opts["mpi_comm"] = COMM;
    
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
                 const Node &data)
{
    // Define Ascent actions

    Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = field_name;
    scenes["s1/renders/r1/image_prefix"] = output_path;

    // Rotate spatial meshes for variety
    if (field_name.find("spatial") != std::string::npos)
    {
        scenes["s1/renders/r1/camera/azimuth"] = 45.0;
    }

    Node actions;

    Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;

    // Execute Ascent actions
    execute_ascent(data, actions);
}

void
render_all_fields(const Node &data,
                  const std::string &output_path,
                  const int &cycle)
{
    // This is here to help identify which ascent execute is throwing an error
    if (0 == par_rank)
    {
        ASCENT_INFO("Executing render_all_fields\n");
    }
    
    const std::vector<std::string> fields {
        "intensities",
        "optical_depth",
        "intensities_spatial",
        "optical_depth_spatial"
    };

    // TODO: Investigate whether we gain any performance by rendering all of these fields with
    // a single set of actions
    for (const auto& field : fields)
    {
        std::string full_output_path = output_path + "_" + field;
        render_blueprint(field, full_output_path, data);
        if (0 == par_rank)
        {
            EXPECT_TRUE(check_test_image(full_output_path, 0.01f, cycle));
        }
    }
}

void
load_and_verify_local_data(Node &data,
                           const std::string &data_path)
{
    conduit::relay::mpi::io::blueprint::load_mesh(data_path, data, COMM);

    Node verify_info;
    EXPECT_TRUE(conduit::blueprint::mpi::mesh::verify(data, verify_info, COMM));
}

const bool
is_vtkm_disabled()
{
    Node n;
    ascent::about(n);
    const bool disabled = "disabled" == n["runtimes/ascent/vtkm/status"].as_string();
    if (0 == par_rank && disabled)
    {
        ASCENT_INFO("Ascent was built without vtkm, skipping test\n");
    }
    return disabled;
}

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
        ASCENT_INFO("Testing x-ray extract using MPI on a conduit braid_uniform_multi_domain example mesh\n");
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
        Node diff_info;
        const bool has_differences = baseline_data.diff(state_output,
                                                        diff_info,
                                                        0.01,
                                                        true);
        if (has_differences)
        {
            ASCENT_INFO("Found differences in the braid_uniform_multi_domain blueprint diff:\n");
            diff_info.print();
        }
        EXPECT_FALSE(has_differences);
    }

    // Render and verify each field
    const int cycle = 0;
    render_all_fields(xray_blueprint_output, query_path, cycle);

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
        ASCENT_INFO("Testing x-ray extract using MPI on a conduit braid_uniform_multi_domain example mesh (rotated)\n");
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
        Node diff_info;
        const bool has_differences = baseline_data.diff(state_output,
                                                        diff_info,
                                                        0.01,
                                                        true);
        if (has_differences)
        {
            ASCENT_INFO("Found differences in the braid_uniform_multi_domain blueprint diff (rotated):\n");
            diff_info.print();
        }
        EXPECT_FALSE(has_differences);
    }

    // Render and verify each field
    const int cycle = 0;
    render_all_fields(xray_blueprint_output, query_path, cycle);

    if (0 == par_rank)
    {
        // Dump info
        std::string msg = "Rendered x-ray diagnostic images using MPI on a conduit braid_uniform_multi_domain example mesh (rotated)";
        ASCENT_ACTIONS_DUMP(actions, query_path, msg);
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
