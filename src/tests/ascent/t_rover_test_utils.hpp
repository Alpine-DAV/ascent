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

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#include <conduit_blueprint_mpi.hpp>
#include <conduit_relay_mpi_io_blueprint.hpp>
#endif

using namespace std;
using namespace conduit;
using namespace ascent;

#ifdef ASCENT_MPI_ENABLED
const MPI_Comm COMM = MPI_COMM_WORLD;
#endif

constexpr int EXAMPLE_MESH_SIDE_DIM = 20;
constexpr int EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM = 11;

//-----------------------------------------------------------------------------
inline void
execute_ascent(const Node& data,
               const Node& actions)
{
    Ascent ascent;
    Node ascent_opts;
#ifdef ASCENT_MPI_ENABLED
    ascent_opts["runtime/type"] = "ascent";
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
render_blueprint(const string &field_name,
                 const string &output_path,
                 const Node &data)
{
    // Define Ascent actions
    Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = field_name;
    scenes["s1/renders/r1/image_prefix"] = output_path;

    // Rotate spatial meshes for test image variety
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

//-----------------------------------------------------------------------------
inline void
render_fields(const Node &data,
              const std::string output_path,
              const int cycle,
              const bool render_intensities = true)
{
    int par_rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm_rank(COMM, &par_rank);
#endif

    // This is here to help identify which ascent execute is throwing an error
    if (0 == par_rank)
    {
        ASCENT_INFO("Executing render_fields\n");
    }
    
    std::vector<std::string> fields {
        "optical_depth",
        "optical_depth_spatial"
    };

    if (render_intensities)
    {
        fields.push_back("intensities");
        fields.push_back("intensities_spatial");
    }

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

//-----------------------------------------------------------------------------
inline void
load_and_verify_local_data(Node &data,
                           const std::string data_path)
{
    Node verify_info;
#ifdef ASCENT_MPI_ENABLED
    conduit::relay::mpi::io::blueprint::load_mesh(data_path, data, COMM);
    EXPECT_TRUE(conduit::blueprint::mpi::mesh::verify(data, verify_info, COMM));
#else
    conduit::relay::io::blueprint::load_mesh(data_path, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
#endif
}

//-----------------------------------------------------------------------------
inline void
load_and_verify_ascent_data(Node &baseline_data,
                            const std::string &filename)
{
    Node verify_info;
    const std::string baseline_path = conduit::utils::join_file_path(std::string(ASCENT_T_DATA_DIR),
                                                                     filename);
#ifdef ASCENT_MPI_ENABLED
    conduit::relay::mpi::io::blueprint::load_mesh(baseline_path, baseline_data, COMM);
    EXPECT_TRUE(conduit::blueprint::mpi::mesh::verify(baseline_data, verify_info, COMM));
#else
    conduit::relay::io::blueprint::load_mesh(baseline_path, baseline_data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(baseline_data, verify_info));
#endif
}

inline void
get_default_baseline(Node &baseline_data,
                     const Node &params,
                     const int &cycle)
{
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
            emission:
            output_type: ""
        xray_data: 
            detector_width: 0.0
            detector_height: 0.0
            intensity_max: 0.0
            intensity_min: 0.0
            optical_depth_max: 0.0
            optical_depth_min: 0.0
            image_topo_order_of_domain_variables: "xyz"
        domain_id: 0
    )yaml";

    // Parse the YAML and then overwrite the test parameters
    baseline_data.parse(yaml);
    baseline_data["cycle"] = cycle;
    baseline_data["xray_query"].update(params["rover"]);

    if (params.has_child("camera"))
    {
        // Camera params are optional, so we only set them if they exist
        baseline_data["xray_query/camera"].set(params["camera"]);
    }
}

//-----------------------------------------------------------------------------
inline void
check_blueprint_diff(const Node &baseline_data,
                     const Node &state_output)
{
    Node diff_info;
    const bool has_differences = baseline_data.diff(state_output,
                                                    diff_info,
                                                    0.01,
                                                    true);

    if (has_differences)
    {
        // Printing the diff is useful for debugging
        ASCENT_INFO("Found differences in the blueprint diff:\n");
        std::cout << diff_info.to_yaml() << std::endl;
    }

    EXPECT_FALSE(has_differences);
}

//-----------------------------------------------------------------------------
inline void
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

//-----------------------------------------------------------------------------
inline void
get_valid_multi_domain_test_data(Node &data, const int num_domains)
{
    for (int i = 0; i < num_domains; i++)
    {
        Node domain;
        conduit::blueprint::mesh::examples::braid("uniform",
                                                  EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM,
                                                  EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM,
                                                  EXAMPLE_MULTI_DOMAIN_MESH_SIDE_DIM,
                                                  domain);

        domain["coordsets/coords/origin/x"] = -10.0 + 20.0 * i;
        domain["state/domain_id"] = i;
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
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));
}

//-----------------------------------------------------------------------------
inline const bool
is_vtkm_disabled()
{
    Node about_info;
    ascent::about(about_info);
    const bool disabled = "disabled" == about_info["runtimes/ascent/vtkm/status"].as_string();

    int par_rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm_rank(COMM, &par_rank);
#endif

    if (0 == par_rank && disabled)
    {
        ASCENT_INFO("Ascent was built without vtkm, skipping test\n");
    }

    return disabled;
}

#endif // T_ROVER_TEST_UTILS_HPP