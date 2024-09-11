//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_render_3d_poly.cpp
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
TEST(ascent_pipeline, test_render_3d_poly)
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
    index_t length = 10;

    conduit::blueprint::mesh::examples::polychain(length, data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_3d_poly");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "chain";
    scenes["s1/image_prefix"] = output_file;

    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));
}

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_3d_poly_multi)
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
    index_t nz = 3;

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
                                                        "tout_render_3d_poly_multi");
    // remove old images before rendering
    remove_test_image(output_file);

    //
    // Create the actions.
    //
    conduit::Node actions;
    conduit::Node &add_plots = actions.append();
    add_plots["action"] = "add_scenes";
    conduit::Node &scenes = add_plots["scenes"];
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = "level";
    scenes["s1/image_prefix"] = output_file;
    actions.print();

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent.open();
    ascent.publish(root);
    ascent.execute(actions);
    ascent.close();

    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));
}

//-----------------------------------------------------------------------------
TEST(ascent_pipeline, test_render_3d_poly_shared_coordset)
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
    // reproducer for issue: https://github.com/Alpine-DAV/ascent/issues/1322
    // generate sides with shared coordset caused name collision with
    // output coordset

    // in this case, having an extra topology with a the same coordset
    // corrupted the rendering of the first topology

    // example mesh
    std::string mesh_yaml= R"EXXAMPLE(
coordsets:
  hex:
    type: "explicit"
    values:
      x: [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
      y: [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
      z: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
topologies:
  hex:
    type: "unstructured"
    coordset: "hex"
    elements:
      shape: "polyhedral"
      connectivity: [0, 1, 2, 3, 4, 5]
      offsets: 0
      sizes: 6
    subelements:
      shape: "polygonal"
      connectivity: [0, 1, 2, 3, 2, 1, 5, 6, 4, 7, 6, 5, 0, 3, 7, 4, 1, 0, 4, 5, 3, 2, 6, 7]
      offsets: [0, 4, 8, 12, 16, 20]
      sizes: [4, 4, 4, 4, 4, 4]
  face2:
    type: "unstructured"
    coordset: "hex"
    elements:
      shape: "polygonal"
      connectivity: [4, 7, 6, 5]
      offsets: 0
      sizes: 4
fields:
  dummy:
    association: "element"
    topology: "hex"
    values: 5.0
)EXXAMPLE";

    Node mesh;
    mesh.parse(mesh_yaml,"yaml");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_render_3d_poly_shared_coordset");
    // remove old images before rendering
    remove_test_image(output_file);

    Node actions;
    Node &pipelines = actions.append();
    pipelines["action"] = "add_pipelines";
    pipelines["pipelines/pl1/f1/type"] = "slice";
    pipelines["pipelines/pl1/f1/params/topology"] = "hex";
    pipelines["pipelines/pl1/f1/params/point/x"] = 0.0;
    pipelines["pipelines/pl1/f1/params/point/y"] = 0.0;
    pipelines["pipelines/pl1/f1/params/point/z"] = 0.5;
    pipelines["pipelines/pl1/f1/params/normal/x"] = 0.0;
    pipelines["pipelines/pl1/f1/params/normal/y"] = 0.0;
    pipelines["pipelines/pl1/f1/params/normal/z"] = 1.0;

    Node &scenes = actions.append();
    scenes["action"] = "add_scenes";
    scenes["scenes/s1/image_prefix"] = "hex_broken";
    scenes["scenes/s1/plots/p1/type"] = "mesh";
    scenes["scenes/s1/plots/p1/topology"] = "hex";
    scenes["scenes/s1/plots/p2/type"] = "pseudocolor";
    scenes["scenes/s1/plots/p2/pipeline"] = "pl1";
    scenes["scenes/s1/plots/p2/field"] = "dummy";
    scenes["scenes/s1/image_prefix"] = output_file;



    //
    // Run Ascent
    //

    Ascent ascent;
    Node ascent_opts;
    ascent.open();
    ascent.publish(mesh);
    ascent.execute(actions);
    ascent.close();

    //
    // // check that we created an image
    EXPECT_TRUE(check_test_image(output_file, 0.001f, "0"));
}
