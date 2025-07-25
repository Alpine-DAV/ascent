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
                  const std::string output_path,
                  const int cycle)
{
    // This is here to help identify which ascent execute is throwing an error
    ASCENT_INFO("Executing render_all_fields\n");
    
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
    const std::string baseline_path = conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),
                                                                     filename);
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

void get_valid_multi_domain_test_data(Node &data, const int num_domains)
{
    for(int i = 0; i < num_domains; i++)
    {
        Node domain;
        conduit::blueprint::mesh::examples::braid("uniform",
                                                  EXAMPLE_MESH_SIDE_DIM,
                                                  EXAMPLE_MESH_SIDE_DIM,
                                                  EXAMPLE_MESH_SIDE_DIM,
                                                  domain);

        domain["coordsets/coords/origin/x"] = -10.0 + 20.0 * i;
        domain["state/domain_id"] = i;
        domain["state/cycle"] = 0;
        domain["fields/rank"].set(domain["fields/radial"]);

        float64_array rank_vals = domain["fields/rank/values"].value();
        for(index_t j = 0; j < rank_vals.number_of_elements(); j++)
        {
            rank_vals[j] = static_cast<float64>(i);
        }

        data.append().set(domain);
    }

    Node verify_info;
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

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

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
    extracts["e1/params/rover/output_type"] = "yaml";
    extracts["e1/params/rover/precision"] = "double";

    Node actions;
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);
    Node &state_output = xray_blueprint_output["domain_000000/state"];

    // Load and verify baseline data
    const std::string yaml = R"yaml(
          time: 3.1414999961853
          cycle: 100
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
            precision: "double"
            width: 200
            unit_scalar: 1.0
            absorption: "radial"
            emission: "radial"
            output_type: "yaml"
          xray_data: 
            detector_width: 4.00000016604152
            detector_height: 4.00000016604152
            intensity_max: 173.205078125
            intensity_min: 0.0
            optical_depth_max: 2698.02783203125
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
        ASCENT_INFO("Found differences in the braid blueprint diff:\n");
        diff_info.print();
    }
    EXPECT_FALSE(has_differences);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of an example braid mesh";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
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

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

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
    extracts["e1/params/rover/output_type"] = "json";
    extracts["e1/params/rover/background_intensity"] = 100.0f;
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
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of an example braid mesh (rotated)";
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
    const std::string query_ext_name = "_000000.cycle_000000.root";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

    // Remove old test image
    const int cycle = 0;
    remove_test_image(query_path, cycle);

    // Generate and verify test data
    Node test_data;
    get_valid_multi_domain_test_data(test_data, 2);

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
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);
    Node &state_output = xray_blueprint_output["domain_000000/state"];

    // Load and verify baseline data
    const std::string yaml = R"yaml(
        time: 3.1414999961853
        cycle: 0
        xray_view: 
            position: [10.0, 3.57627868652344e-07, 48.9897956848145]
            zoom: 1.0
            look_at: [10.0, 3.57627868652344e-07, 3.57627868652344e-07]
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
            optical_depth_max: 3616.85577975963
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

    // Render and verify each field
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered xray diagnostic images of an example braid_uniform_multi_domain mesh";
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
    const std::string query_ext_name = "_000000.cycle_000000.root";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

    // Remove old test image
    const int cycle = 0;
    remove_test_image(query_path, cycle);

    // Generate and verify test data
    Node test_data;
    get_valid_multi_domain_test_data(test_data, 2);

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
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);
    Node &state_output = xray_blueprint_output["domain_000000/state"];

    // Load and verify baseline data
    const std::string yaml = R"yaml(
        time: 3.1414999961853
        cycle: 0
        xray_view: 
            position: [40.0, 34.6410140991211, 17.3205070495605]
            zoom: 1.0
            look_at: [10.0, 3.57627868652344e-07, 3.57627868652344e-07]
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
            optical_depth_max: 3152.70922851562
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

    // Render and verify each field
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered xray diagnostic images of an example braid_uniform_multi_domain mesh (rotated)";
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
    const std::string query_ext_name = "_000048.cycle_000048.root";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

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
    extracts["e1/params/rover/output_type"] = "yaml";
    extracts["e1/params/rover/precision"] = "double";

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Load and verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);
    Node &state_output = xray_blueprint_output["domain_000000/state"];

    // Load and verify baseline data
    const std::string yaml = R"(
        time: 4.80000019073486
        cycle: 48
        xray_view: 
          position: [0.0, 2.5, 47.0156211853027]
          zoom: 1.0
          look_at: [0.0, 2.5, 15.0]
          up: [0.0, 1.0, 0.0]
          fov: 60.0
          xpan: 0.0
          ypan: 0.0
          near_plane: 3.20156216621399
          far_plane: 320.156219482422
        xray_query: 
          background_intensity: 0.0
          divide_emis_by_absorb: "false"
          enable_rays_mesh: "false"
          height: 200
          precision: "double"
          width: 200
          unit_scalar: 1.0
          absorption: "d"
          emission: "p"
          output_type: "yaml"
        xray_data: 
          detector_width: 3.69684552235394
          detector_height: 3.69684552235394
          intensity_max: 0.491446942090988
          intensity_min: 0.0
          optical_depth_max: 125.497886657715
          optical_depth_min: 0.0
          image_topo_order_of_domain_variables: "xyz"
        domain_id: 0
        )";

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
        ASCENT_INFO("Found differences in the curv3d blueprint diff:\n");
        diff_info.print();
    }
    EXPECT_FALSE(has_differences);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the curv3d dataset";
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
    const std::string query_ext_name = "_000048.cycle_000048.root";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

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
    extracts["e1/params/rover/output_type"] = "json";
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
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the curv3d dataset (rotated)";
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
    const std::string query_ext_name = "_000048.cycle_000048.root";

    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

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
    extracts["e1/params/rover/output_type"] = "hdf5";

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
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the curv3d dataset (all camera params)";
    ASCENT_ACTIONS_DUMP(actions, query_path, msg);
}

// TODO: Add a test for imaging planes and the rays mesh

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
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

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
    extracts["e1/params/rover/output_type"] = "yaml";
    // TODO: Investigate why using double precision with this
    // dataset has an artifact in the intensity output
    // extracts["e1/params/rover/precision"] = "double";
    extracts["e1/params/rover/divide_emis_by_absorb"] = "true";

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;

    // Execute Ascent actions
    execute_ascent(test_data, actions);

    // Verify output mesh
    Node xray_blueprint_output, verify_info;
    load_and_verify_local_data(xray_blueprint_output, output_data_path);
    Node &state_output = xray_blueprint_output["domain_000000/state"];

    // Load and verify baseline data
    const std::string yaml = R"(
        time: 4.80000019073486
        cycle: 48
        xray_view:
          position: [0.0, 2.49999904632568, 47.0156211853027]
          zoom: 1.0
          look_at: [0.0, 2.49999904632568, 15.0]
          up: [0.0, 1.0, 0.0]
          fov: 60.0
          xpan: 0.0
          ypan: 0.0
          near_plane: 3.20156216621399
          far_plane: 320.156219482422
        xray_query:
          background_intensity: 0.0
          divide_emis_by_absorb: "true"
          enable_rays_mesh: "false"
          height: 200
          precision: "single"
          width: 200
          unit_scalar: 1.0
          absorption: "d"
          emission: "p"
          output_type: "yaml"
        xray_data:
          detector_width: 3.69684552235394
          detector_height: 3.69684552235394
          intensity_max: 0.241532012820244
          intensity_min: 0.0
          optical_depth_max: 125.49796295166
          optical_depth_min: 0.0
          image_topo_order_of_domain_variables: "xyz"
        domain_id: 0
        )";

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
        ASCENT_INFO("Found differences in the curv3d blueprint diff:\n");
        diff_info.print();
    }
    EXPECT_FALSE(has_differences);

    // Render and verify each field
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the multi_curv3d dataset";
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
    const std::string query_ext_name = "_000048.cycle_000048.root";
    
    // Setup paths
    const std::string output_path = prepare_output_dir();
    const std::string query_path = conduit::utils::join_file_path(output_path, 
                                                                  query_name);
    const std::string output_data_path = query_path + query_ext_name;

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
    extracts["e1/params/rover/output_type"] = "json";
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
    render_all_fields(xray_blueprint_output, query_path, cycle);

    // Dump info
    std::string msg = "Rendered XRay diagnostic images of the multi_curv3d dataset (rotated)";
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
