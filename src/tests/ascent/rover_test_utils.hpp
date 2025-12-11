//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_rover_test_utils.hpp
///
//-----------------------------------------------------------------------------
#ifndef T_ROVER_TEST_UTILS_HPP
#define T_ROVER_TEST_UTILS_HPP

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_utils.hpp"

#ifdef ROVER_TEST_MPI_ENABLED
#include <mpi.h>
#include <conduit_blueprint_mpi.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>
#endif

using namespace std;
using namespace conduit;
using namespace ascent;

#ifdef ROVER_TEST_MPI_ENABLED
const MPI_Comm COMM = MPI_COMM_WORLD;
#endif

constexpr int EXAMPLE_MESH_SIDE_DIM = 20;
constexpr int EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM = 11;

//-----------------------------------------------------------------------------
inline void
execute_ascent(const Node &data,
               const Node &actions)
{
    Ascent ascent;
    Node ascent_opts;
    ascent_opts["exceptions"] = "forward";
#ifdef ROVER_TEST_MPI_ENABLED
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(COMM);
#endif
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    // TODO can we ask Ascent for the name of the file it wrote?
    // std::cout << ascent.info().to_yaml() << std::endl;
    ascent.close();
}

//-----------------------------------------------------------------------------
inline void
render_fields(const Node &data,
              const std::string &output_path,
              const int cycle = 0,
              const bool render_intensities = true)
{
    int par_rank = 0;
#ifdef ROVER_TEST_MPI_ENABLED
    MPI_Comm_rank(COMM, &par_rank);
#endif
    const bool is_root = (0 == par_rank);

    // This is here to help identify which ascent execute is throwing an error
    if (is_root)
    {
        ASCENT_INFO("Executing render_fields\n");
    }
    
    std::vector<std::string> fields {
        "optical_depth",
        "optical_depth_spatial"
    };

    // We won't have intensities in the absorption-only case
    if (render_intensities)
    {
        fields.push_back("intensities");
        fields.push_back("intensities_spatial");
    }

    // Render all fields with a single set of actions
    Node scenes;
    Node actions;
    Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";

    // Create scenes for all fields
    for (int i = 0; i < fields.size(); i++)
    {
        std::string field = fields[i];
        std::string scene_name = "s" + std::to_string(i + 1);
        std::string plot_name = "p" + std::to_string(i + 1);
        std::string render_name = "r" + std::to_string(i + 1);
        std::string full_output_path = output_path + "_" + field;

        scenes[scene_name]["plots"][plot_name]["type"] = "pseudocolor";
        scenes[scene_name]["plots"][plot_name]["field"] = field;
        scenes[scene_name]["renders"][render_name]["image_prefix"] = full_output_path;

        // Rotate spatial meshes for test image variety
        if (field.find("spatial") != std::string::npos)
        {
            scenes[scene_name]["renders"][render_name]["camera/azimuth"] = 45.0;
        }
    }

    add_plots["scenes"] = scenes;

    // Execute all renders in a single Ascent call for better performance
    execute_ascent(data, actions);

    // Check all generated images
    if (is_root)
    {
        for (const auto& field : fields)
        {
            std::string full_output_path = output_path + "_" + field;
            EXPECT_TRUE(check_test_image(full_output_path, 0.01f, cycle));
        }
    }
}

//-----------------------------------------------------------------------------
inline void
render_multi_group_fields(const Node &data,
                          const std::string &output_path,
                          const int cycle = 0,
                          const bool render_intensities = true)
{
    int par_rank = 0;
#ifdef ROVER_TEST_MPI_ENABLED
    MPI_Comm_rank(COMM, &par_rank);
#endif
    const bool is_root = (0 == par_rank);

    // This is here to help identify which ascent execute is throwing an error
    if (is_root)
    {
        ASCENT_INFO("Executing render_multi_group_fields\n");
    }
    
    std::vector<std::string> fields {
        "optical_depth",
        "optical_depth_spatial"
    };

    // We won't have intensities in the absorption-only case
    if (render_intensities)
    {
        fields.push_back("intensities");
        fields.push_back("intensities_spatial");
    }

    // Create slices at the midpoints between different z values
    std::vector<double> z_values = {0.5, 1.5, 2.5};

    Node pipelines;
    Node scenes;
    Node actions;

    // Add pipelines action
    Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";

    // Add scenes action
    Node &add_scenes = actions.append();
    add_scenes["action"] = "add_scenes";

    int counter = 1;

    // Create pipelines and scenes for all fields
    for (const auto& field : fields)
    {
        const bool is_spatial_mesh = field.find("spatial") != std::string::npos;

        // Full topology render
        std::string scene_name = "s" + std::to_string(counter);
        std::string plot_name = "p" + std::to_string(counter);
        std::string render_name = "r" + std::to_string(counter);
        std::string full_output_path = output_path + "_" + field + "_full";

        // Create scene with pseudocolor plot (no pipeline needed for full topology)
        scenes[scene_name]["plots"][plot_name]["type"] = "pseudocolor";
        scenes[scene_name]["plots"][plot_name]["field"] = field;
        scenes[scene_name]["renders"][render_name]["image_prefix"] = full_output_path;

        // Rotate and elevate 3D meshes
        scenes[scene_name]["renders"][render_name]["camera/azimuth"] = 45.0;
        scenes[scene_name]["renders"][render_name]["camera/elevation"] = 45.0;

        counter += 1;

        // Slice renders at different z values
        for (int i = 0; i < z_values.size(); i++)
        {
            double z_val = z_values[i];
            std::string pipeline_name = "pl" + std::to_string(counter);
            std::string scene_name = "s" + std::to_string(counter);
            std::string plot_name = "p" + std::to_string(counter);
            std::string render_name = "r" + std::to_string(counter);
            std::string full_output_path = output_path + "_" + field + "_z" + std::to_string(i);

            // Create pipeline with slice filter
            std::string topology = "image_topo";
            if (is_spatial_mesh)
            {
                topology = "spatial_topo";
            }
            
            pipelines[pipeline_name]["f1"]["type"] = "slice";
            pipelines[pipeline_name]["f1"]["params"]["topology"] = topology;
            pipelines[pipeline_name]["f1"]["params"]["point"]["x"] = 0.0;
            pipelines[pipeline_name]["f1"]["params"]["point"]["y"] = 0.0;
            pipelines[pipeline_name]["f1"]["params"]["point"]["z"] = z_val;
            pipelines[pipeline_name]["f1"]["params"]["normal"]["x"] = 0.0;
            pipelines[pipeline_name]["f1"]["params"]["normal"]["y"] = 0.0;
            pipelines[pipeline_name]["f1"]["params"]["normal"]["z"] = 1.0;

            // Create scene with pseudocolor plot
            scenes[scene_name]["plots"][plot_name]["type"] = "pseudocolor";
            scenes[scene_name]["plots"][plot_name]["field"] = field;
            scenes[scene_name]["plots"][plot_name]["pipeline"] = pipeline_name;
            scenes[scene_name]["renders"][render_name]["image_prefix"] = full_output_path;
            
            counter += 1;
        }
    }

    add_pipelines["pipelines"] = pipelines;
    add_scenes["scenes"] = scenes;

    // Execute all renders in a single Ascent call for better performance
    execute_ascent(data, actions);

    // Check all generated images
    if (is_root)
    {
        for (const auto& field : fields)
        {
            // Full topology
            std::string full_output_path = output_path + "_" + field + "_full";
            EXPECT_TRUE(check_test_image(full_output_path, 0.01f, cycle));

            // Slices
            for (int i = 0; i < z_values.size(); i++)
            {
                std::string full_output_path = output_path + "_" + field + "_z" + std::to_string(i);
                EXPECT_TRUE(check_test_image(full_output_path, 0.01f, cycle));
            }
        }
    }
}

//-----------------------------------------------------------------------------
inline void
load_and_verify_local_data(Node &data,
                           const std::string &data_path)
{
    Node verify_info;
#ifdef ROVER_TEST_MPI_ENABLED
    relay::mpi::io::blueprint::load_mesh(data_path, data, COMM);
    EXPECT_TRUE(blueprint::mpi::mesh::verify(data, verify_info, COMM));
#else
    relay::io::blueprint::load_mesh(data_path, data);
    EXPECT_TRUE(blueprint::mesh::verify(data, verify_info));
#endif
}

//-----------------------------------------------------------------------------
inline void
load_and_verify_ascent_data(Node &baseline_data,
                            const std::string &filename)
{
    Node verify_info;
    const std::string baseline_path = utils::join_file_path(
                                          std::string(ASCENT_T_DATA_DIR),
                                          filename
                                      );
#ifdef ROVER_TEST_MPI_ENABLED
    relay::mpi::io::blueprint::load_mesh(baseline_path, baseline_data, COMM);
    EXPECT_TRUE(blueprint::mpi::mesh::verify(baseline_data, verify_info, COMM));
#else
    relay::io::blueprint::load_mesh(baseline_path, baseline_data);
    EXPECT_TRUE(blueprint::mesh::verify(baseline_data, verify_info));
#endif
}

inline void
get_default_baseline(Node &baseline_data,
                     const Node &params,
                     const int cycle = 0)
{
    // The test params that we just used to generate the output
    const Node &xray_query = params["rover"];

    // Baseline data with default values
    const std::string yaml = R"yaml(
        time: 0.0
        cycle: 0
        xray_view: 
            position: [0.0, 0.0, 0.0]
            zoom: 1.0
            look_at: [0.0, 0.0, 0.0]
            up: [0.0, 1.0, 0.0]
            fov: 60.0
            xpan: 0.0
            ypan: 0.0
            near_plane: 0.0
            far_plane: 0.0
        xray_query: 
            background_intensity: 0.0
            divide_emis_by_absorb: "false"
            enable_rays_mesh: "false"
            height: 200
            precision: "single"
            width: 200
            unit_scalar: 1.0
            absorption: ""
            output_type: ""
        xray_data: 
            detector_width: 0.0
            detector_height: 0.0
            optical_depth_max: 0.0
            optical_depth_min: 0.0
            image_topo_order_of_domain_variables: "xyz"
        domain_id: 0
    )yaml";

    // Parse the YAML and then overwrite the test parameters
    baseline_data.parse(yaml);
    baseline_data["cycle"] = cycle;
    baseline_data["xray_query"].update(xray_query);

    // Camera params are optional, so we only set them if they exist
    if (params.has_child("camera"))
    {
        baseline_data["xray_query/camera"].set(params["camera"]);
    }

    // There are 2 cases to support with respect to the emission field:
    // 1. Emission is not set in the params: the absorption-only case
    // 2. Emission is set to a non-empty string: the absorption + emission case
    if (xray_query.has_child("emission"))
    {
        baseline_data["xray_data/intensity_max"] = 0.0;
        baseline_data["xray_data/intensity_min"] = 0.0;
    }
}

//-----------------------------------------------------------------------------
inline void
check_blueprint_diff(const Node &baseline_data,
                     const Node &state_output)
{
    Node diff_info;
    const double epsilon = 0.01;
    const bool relaxint = true;
    const bool has_differences = baseline_data.diff(state_output,
                                                    diff_info,
                                                    epsilon,
                                                    relaxint);

    if (has_differences)
    {
        // Printing the diff is useful for debugging
        ASCENT_INFO("Found differences in the blueprint diff:\n");
        ASCENT_INFO("\n===================BASELINE DATA BEGIN===================");
        ASCENT_INFO(baseline_data.to_yaml());
        ASCENT_INFO("===================BASELINE DATA END===================\n");
        ASCENT_INFO("\n===================OUTPUT DATA BEGIN===================");
        ASCENT_INFO(state_output.to_yaml());
        ASCENT_INFO("===================OUTPUT DATA END===================\n");
        ASCENT_INFO("\n===================DIFF RESULTS BEGIN===================");
        ASCENT_INFO(diff_info.to_yaml());
        ASCENT_INFO("===================DIFF RESULTS END===================\n\n");
    }

    EXPECT_FALSE(has_differences);
}

inline void
add_camera_rotation(Node &extracts,
                    const double azimuth = 45.0,
                    const double elevation = 45.0)
{
    extracts["e1/params/camera/azimuth"] = azimuth;
    extracts["e1/params/camera/elevation"] = elevation;
}

//-----------------------------------------------------------------------------
inline void
get_braid_test_data(Node &data)
{
    Node verify_info;
    blueprint::mesh::examples::braid("hexs",
                                     EXAMPLE_MESH_SIDE_DIM,
                                     EXAMPLE_MESH_SIDE_DIM,
                                     EXAMPLE_MESH_SIDE_DIM,
                                     data);
    EXPECT_TRUE(blueprint::mesh::verify(data, verify_info));
}

//-----------------------------------------------------------------------------
inline void
get_braid_multi_domain_test_data(Node &data,
                                 const int num_domains = 2,
                                 const int override = 0)
{
    for (int i = 0; i < num_domains; i++)
    {
        Node domain;
        blueprint::mesh::examples::braid("uniform",
                                         EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM,
                                         EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM,
                                         EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM,
                                         domain);

        if (0 == override)
        {
            // Normal case, where one rank wants valid multi-domain data
            domain["coordsets/coords/origin/x"] = -10.0 + 20.0 * i;
            domain["state/domain_id"] = i;
        }
        else // (0 != override)
        {
            // Override case, where one rank wants to generate its own domain,
            // but we don't want to call the MPI utility
            domain["coordsets/coords/origin/x"] = -10.0 + 20.0 * override;
            domain["state/domain_id"] = override;
        }

        domain["state/cycle"] = 0;
        domain["fields/rank"].set(domain["fields/radial"]);

        float64_array rank_vals = domain["fields/rank/values"].value();
        for (int j = 0; j < rank_vals.number_of_elements(); j++)
        {
            rank_vals[j] = static_cast<float64>(i);
        }

        data.append().set(domain);
    }

    Node verify_info;
    EXPECT_TRUE(blueprint::mesh::verify(data, verify_info));
}

//-----------------------------------------------------------------------------
#ifdef ROVER_TEST_MPI_ENABLED
inline void
get_mpi_braid_multi_domain_test_data(Node &data,
                                     const int par_rank,
                                     const int par_size)
{
    Node verify_info;

    // All of rover's baseline images are generated with 2 MPI ranks
    // (i.e. only 2 domains in the input), so we need to ensure that the test
    // data always matches that case regardless of the number of MPI ranks that
    // are used to execute the test. Furthermore, separating the cases like this
    // allows us to make some per-case optimizations
    if (2 == par_size)
    {
        // This is the default case that we'll have in CI, 2 MPI ranks, so
        // we check for it first. We use the MPI utility since this is a case in
        // which we know we want every rank to participate
        blueprint::mpi::mesh::examples::braid_uniform_multi_domain(data, COMM);

        // Verify test data
        EXPECT_TRUE(conduit::blueprint::mpi::mesh::verify(data, verify_info, COMM));
    }
    else if (1 == par_size)
    {
        // If we're executing with 1 MPI rank, it needs to get both domains, so we can
        // just call the same utility as the serial case
        get_braid_multi_domain_test_data(data);
    }
    else if (par_rank <= 1)
    {
        // If we're executing with more than 2 MPI ranks, we only want to generate
        // data on ranks 0 and 1. We avoid the MPI utility in this case because we
        // would have to call it with every rank and then subsequently call .reset()
        // on ranks > 1, which is wasteful
        get_braid_multi_domain_test_data(data, 1, par_rank);

        // Verify test data
        EXPECT_TRUE(blueprint::mesh::verify(data, verify_info));
    }
}
#endif

//-----------------------------------------------------------------------------
inline void
get_multi_group_curv3d_data(Node &data)
{
    // Load dataset and verify blueprint
    const std::string &filename = "multi_curv3d_blueprint.cycle_000048.root";
    load_and_verify_ascent_data(data, filename);

    // Build multi-group fields per domain
    for (const auto &child : data.child_names())
    {
        int64 num_elements = data[child]["fields/d/values"].dtype().number_of_elements();

        // Allocate new arrays
        data[child]["fields/d_multi/values/d0"].set(DataType::float32(num_elements));
        data[child]["fields/d_multi/values/d1"].set(DataType::float32(num_elements));
        data[child]["fields/d_multi/values/d2"].set(DataType::float32(num_elements));
        data[child]["fields/p_multi/values/p0"].set(DataType::float32(num_elements));
        data[child]["fields/p_multi/values/p1"].set(DataType::float32(num_elements));
        data[child]["fields/p_multi/values/p2"].set(DataType::float32(num_elements));

        // Get array accessors
        float32_array d_vals  = data[child]["fields/d/values"].value();
        float32_array p_vals  = data[child]["fields/p/values"].value();
        float32_array d0_vals = data[child]["fields/d_multi/values/d0"].value();
        float32_array d1_vals = data[child]["fields/d_multi/values/d1"].value();
        float32_array d2_vals = data[child]["fields/d_multi/values/d2"].value();
        float32_array p0_vals = data[child]["fields/p_multi/values/p0"].value();
        float32_array p1_vals = data[child]["fields/p_multi/values/p1"].value();
        float32_array p2_vals = data[child]["fields/p_multi/values/p2"].value();

        // Create additional fields by scaling existing ones
        for (int i = 0; i < num_elements; i++)
        {
            d0_vals[i] = d_vals[i];
            d1_vals[i] = d_vals[i] * 6.0f;
            d2_vals[i] = d_vals[i] * 3.0f;

            p0_vals[i] = p_vals[i];
            p1_vals[i] = p_vals[i] * 6.0f;
            p2_vals[i] = p_vals[i] * 3.0f;
        }

        // Set topology and association for new fields
        data[child]["fields/d_multi/topology"] = "mesh1";
        data[child]["fields/p_multi/topology"] = "mesh1";
        data[child]["fields/d_multi/association"] = "element";
        data[child]["fields/p_multi/association"] = "element";
    }

    // Verify test data
    Node verify_info;
#ifdef ROVER_TEST_MPI_ENABLED
    EXPECT_TRUE(blueprint::mpi::mesh::verify(data, verify_info, COMM));
#else
    EXPECT_TRUE(blueprint::mesh::verify(data, verify_info));
#endif
}

//-----------------------------------------------------------------------------
inline void
remove_rover_test_data(const std::string &path,
                       const std::string &suffix,
                       const int cycle = 0)
{
    const std::vector<std::string> fields {
        "optical_depth",
        "optical_depth_spatial",
        "intensities",
        "intensities_spatial"
    };

    // Remove old test images
    for (const auto& field : fields)
    {
        std::string image_path = path + "_" + field;
        remove_test_image(image_path, cycle);
    }

    // Also remove the main output file
    std::string output_file = path + suffix;
    remove_test_file(output_file);
}

//-----------------------------------------------------------------------------
inline void
get_default_action_params(Node &actions,
                          const Node &extracts)
{
    Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    add_extracts["extracts"] = extracts;
}

//-----------------------------------------------------------------------------
inline void
get_common_extract_params(Node &extracts,
                           const std::string &query_path,
                           const std::string &absorption,
                           const std::string &emission,
                           const std::string &output_type = "yaml")
{
    extracts["e1/type"] = "xray";
    extracts["e1/params/rover/filename"] = query_path;
    extracts["e1/params/rover/absorption"] = absorption;
    if (!emission.empty())
    {
        // We don't support the case where emission is set to an empty string,
        // so we only set it if it's non-empty
        extracts["e1/params/rover/emission"] = emission;
    }
    extracts["e1/params/rover/output_type"] = output_type;
}

//-----------------------------------------------------------------------------
inline const bool
is_viskores_disabled(const bool is_root = true)
{
    Node about_info;
    ascent::about(about_info);
    if ("disabled" != about_info["runtimes/ascent/viskores/status"].as_string())
    {
        // Viskores is enabled
        return false;
    }

    if (is_root)
    {
        ASCENT_INFO("Skipping test: Ascent was built without Viskores\n");
    }

    // Viskores is disabled
    return true;
}

#endif // T_ROVER_TEST_UTILS_HPP