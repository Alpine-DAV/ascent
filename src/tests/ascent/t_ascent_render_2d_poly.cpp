//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_2d_poly.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using namespace ascent;
//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_2d_poly)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }
    
    //
    // Create example mesh.
    //
    Node data, verify_info;
    index_t nlevels = 10;
    index_t nz = 1;

    conduit::blueprint::mesh::examples::polytess(nlevels, nz, data);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_2d_poly");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "level";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    Node ascent_info;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(ascent_info);
    EXPECT_EQ(ascent_info["runtime/type"].as_string(), "ascent");
    ascent_info.print();
    ascent.close();

    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));
}

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_2d_poly_multi)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    //
    // Create example mesh.
    //
    Node root, verify_info;
    Node &child1 = root.append();
    Node &child2 = root.append();
    index_t nlevels = 3;
    index_t nz = 1;

    conduit::blueprint::mesh::examples::polytess(nlevels, nz, child1);
    conduit::blueprint::mesh::examples::polytess(nlevels, nz, child2);

    EXPECT_TRUE(conduit::blueprint::mesh::verify(child1, verify_info));
    EXPECT_TRUE(conduit::blueprint::mesh::verify(child2, verify_info));

    float64 *y_values = child2["coordsets/coords/values/y"].value();

    const int num_elements = child1["coordsets/coords/values/y"].dtype().number_of_elements();

    for (int i = 0; i < num_elements; i ++)
    {
        y_values[i] += 10.0f;
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(child2, verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_2d_poly_multi");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "level";
    scenes["s1/image_prefix"] = output_file;

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    Node ascent_info;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(root);
    ascent.execute(actions);
    ascent.info(ascent_info);
    EXPECT_EQ(ascent_info["runtime/type"].as_string(), "ascent");
    ascent_info.print();
    ascent.close();

    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));
}

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_2d_poly_and_nonpoly)
{
    // the vtkm runtime is currently our only rendering runtime
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
        return;
    }

    //
    // Create example mesh.
    //
    Node data, braid_data, verify_info;
    index_t nlevels = 5;
    index_t nz = 1;

    conduit::blueprint::mesh::examples::polytess(nlevels, nz, data);

    conduit::blueprint::mesh::examples::braid("quads", 5,5,0,braid_data);

    data["coordsets/coords_braid"] = braid_data["coordsets/coords"];
    data["topologies/mesh"] = braid_data["topologies/mesh"];
    data["topologies/mesh/coordset"] = "coords_braid";
    data["fields/braid"] = braid_data["fields/braid"];
    data["fields/radial"] = braid_data["fields/radial"];
    data["fields/vel"] = braid_data["fields/vel"];

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_2d_poly_and_nopoly");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "level";
    scenes["s1/image_prefix"] = output_file + "polytess";
    scenes["s2/plots/p1/type"] = "pseudocolor";
    scenes["s2/plots/p1/field"] = "braid";
    scenes["s2/image_prefix"] = output_file + "braid";

    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    add_plots["scenes"] = scenes;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    Node ascent_info;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.info(ascent_info);
    EXPECT_EQ(ascent_info["runtime/type"].as_string(), "ascent");
    ascent_info.print();
    ascent.close();

    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file + "polytess", 0.001f, "0"));
    EXPECT_TRUE(check_test_image(output_file + "braid", 0.001f, "0"));
}
