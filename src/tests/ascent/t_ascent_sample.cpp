//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_uniform_grid.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"




using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
TEST(ascent_sample, line_2d)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("quads",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              0,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a 3d line of points");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_sample_line_2d");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //
    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          line:
            num_samples: 100
            start:
              x: 1.0
              y: 1.0
            end:
              x: 0.0
              y: 0.0
          invalid_value: -10.0
- 
  action: "add_extracts"
  extracts: 
    e1:
      pipeline: pl1
      type: "relay"
      params:
        protocol: "blueprint/mesh/yaml"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["extracts/e1/params/path"] = output_file;
    //actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // // check that we created an image
    // EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter to sample points along a 2d line.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_sample, line_3d)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a 3d line of points");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_sample_line_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //
    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          line:
            num_samples: 100
            start:
              x: 1.0
              y: 1.0
              z: 1.0
            end:
              x: 0.0
              y: 0.0
              z: 0.0
          invalid_value: -10.0
- 
  action: "add_extracts"
  extracts: 
    e1:
      pipeline: pl1
      type: "relay"
      params:
        protocol: "hdf5"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["extracts/e1/params/path"] = output_file;
    //actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // // check that we created an image
    // EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter to sample points along a 3d line.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}


//-----------------------------------------------------------------------------
TEST(ascent_sample, points_2d)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling at a list of 2d points");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_sample_pts_2d");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //
    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid","radial"]
          points:
            x: [-9.0, 0.0, 3.0, 0.0, 3.0, -5.0, 7.24, -7.24, 9.0]
            y: [-9.0, 0.0, 3.0, 3.0, 0.0, -5.0, -8.34, 8.34, 9.0]
          invalid_value: -10.0
- 
  action: "add_extracts"
  extracts: 
    e1:
      pipeline: pl1
      type: "relay"
      params:
        protocol: "hdf5"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["extracts/e1/params/path"] = output_file;
    //actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // // check that we created an image
    // EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter to sample a list of 2d points.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_sample, points_3d)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling at a list of 3d points");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_sample_pts_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //
    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid","radial"]
          points:
            x: [-9.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0, -5.0, 7.24, -7.24, 9.0]
            y: [-9.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 0.0, -5.0, -8.34, 8.34, 9.0]
            z: [-9.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0, -5.0, 4.78,  4.78, 9.0]
          invalid_value: -10.0
- 
  action: "add_extracts"
  extracts: 
    e1:
      pipeline: pl1
      type: "relay"
      params:
        protocol: "hdf5"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["extracts/e1/params/path"] = output_file;
    //actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // // check that we created an image
    // EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter to sample a list of 3d points.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, box_3d)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a 3D box");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_sample_box_3d");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //
    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          box:
            dims:
              i: 5.0
              j: 5.0
              k: 5.0
            max:
              x: max 
              y: max 
              z: max 
            min:
              x: 0.0
              y: 0.0
              z: 0.0
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1:
      plots:
        p1:
          type: "pseudocolor"
          field: "braid"
          pipeline: pl1
- 
  action: "add_extracts"
  extracts: 
    e1:
      pipeline: pl1
      type: "relay"
      params:
        protocol: "hdf5"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/image_prefix"] = output_file;
    actions[2]["extracts/e1/params/path"] = output_file;
    //actions.print();

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    // // check that we created an image
    // EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter to sample points 3d box.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_sample, plane)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a plane");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_sample_plane");

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //
    std::string acts_str = R"xyzxyz(
-
  action: "add_pipelines"
  pipelines:
    pl1:
      f1:
        type: "sample"
        params:
          fields: ["braid"]
          plane:
            point:
              x: 0.0
              y: 0.0
              z: 0.0
            normal:
              x: 0.0
              y: 1.0
              z: 0.0
            dims:
              i: 5.0
              k: 5.0
            spacing:
              dx: 1.0
              dz: 1.0
          invalid_value: -10.0
-
  action: "add_extracts"
  extracts:
    e1:
      pipeline: pl1
      type: "relay"
      params:
        protocol: "hdf5"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["extracts/e1/params/path"] = output_file;

    //
    // Run Ascent
    //
    Ascent ascent;
    ascent.open();
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();

    std::string msg = "An example of using the sample filter to sample points on a plane.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------

TEST(ascent_sample, test_uniform_grid_slice_along_y)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a smaller regular grid of hexahedron input");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_sample_in_y");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //
    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              i: 10
              j: 0
              k: 10
            origin: 
              x: -10 
              y: -10 
              z: -10 
            spacing:
              dx: 1
              dy: 1
              dz: 1
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
      renders: 
        r1: 
          camera: 
            elevation: 30
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter sampling along y (y=0).";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_sample_along_x)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a smaller regular grid of hexahedron input");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_sample_in_x");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              i: 0
              j: 10
              k: 10
            origin: 
              x: -10 
              y: -10 
              z: -10 
            spacing:
              dx: 1
              dy: 1
              dz: 1
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
      renders: 
        r1: 
          camera: 
            azimuth: 90
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter sampling along x (x=0).";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_smaller_in_i)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a smaller regular grid of hexahedron input");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_smaller_in_i");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              i: 10
            spacing:
              dx: 1
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with a smaller x dim.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_smaller_in_j)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a smaller regular grid of hexahedron input");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_smaller_in_j");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              j: 10
            spacing:
              dy: 1
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with a smaller j.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_smaller_in_k)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a smaller regular grid of hexahedron input");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_smaller_in_k");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          field: "braid"
          uniform_grid:
            dims:
              k: 10
            spacing:
              dz: 1
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
      renders: 
        r1: 
          camera: 
            azimuth: 90
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with a smaller k.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_smaller_by10_than_input)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a smaller regular grid of hexahedron input");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_smaller_by10_grid");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              i: 10
              j: 10
              k: 10
            origin: 
              x: -10 
              y: -10 
              z: -10 
            spacing:
              dx: 1
              dy: 1
              dz: 1
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with smaller dims.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_equal_size_input)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_sample_input_dims");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              i: 20
              j: 20
              k: 20
            origin: 
              x: -10 
              y: -10 
              z: -10 
            spacing:
              dx: 1
              dy: 1
              dz: 1
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with dims equal to the input dims.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_equal_size_input_increased_spacing)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_equal_dims_increase_spacing");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            spacing:
              dx: 2.0
              dy: 2.0
              dz: 2.0
          invalid_value: -10.0
      f2: 
        type: "slice"
        params: 
          point: 
            x: 0.0
            y: 0.0
            z: 0.0
          normal: 
            x: 0.0
            y: 0.0
            z: 1.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with increased spacing.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_equal_size_input_decreased_spacing)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_equal_dims_decrease_spacing");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            spacing:
              dx: 0.5
              dy: 0.5
              dz: 0.5
          invalid_value: -10.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with decreased spacing.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_equal_size_input_shift_origin)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_shift_origin");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            origin:
              x: -5.0
              y: -5.0
              z: -5.0
          invalid_value: -10.0
      f2: 
        type: "slice"
        params: 
          point: 
            x: 0.0
            y: 0.0
            z: 0.0
          normal: 
            x: 0.0
            y: 0.0
            z: 1.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with an origin shift.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_equal_size_input_shift_origin_x)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_shift_origin_x");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            origin:
              x: 0.0
          invalid_value: -10.0
      f2: 
        type: "slice"
        params: 
          point: 
            x: 0.0
            y: 0.0
            z: 0.0
          normal: 
            x: 0.0
            y: 0.0
            z: 1.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter shifting the origin along x.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_equal_size_input_shift_origin_y)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_shift_origin_y");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            origin:
              y: 0.0
          invalid_value: -10.0
      f2: 
        type: "slice"
        params: 
          point: 
            x: 0.0
            y: 0.0
            z: 0.0
          normal: 
            x: 0.0
            y: 0.0
            z: 1.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter shifting the origin along y.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_equal_size_input_shift_origin_z)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_shift_origin_z");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            origin:
              z: 0.0
          invalid_value: -10.0
      f2: 
        type: "slice"
        params: 
          point: 
            x: 0.0
            y: 0.0
            z: 0.0
          normal: 
            x: 0.0
            y: 0.0
            z: 1.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter shifting the origin along z.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_larger_by5_than_input)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a larger regular grid of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_larger_by5_grid");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              i: 25
              j: 25
              k: 25
            origin: 
              x: -10 
              y: -10 
              z: -10 
            spacing:
              dx: 1
              dy: 1
              dz: 1
          invalid_value: -10.0
      f2: 
        type: "slice"
        params: 
          point: 
            x: 0.0
            y: 0.0
            z: 0.0
          normal: 
            x: 0.0
            y: 0.0
            z: 1.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with larger dimensions.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_larger_by5_than_input_large_invalid_value)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a larger regular grid of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_larger_by5_grid_with_invalid_value");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
            dims:
              i: 25
              j: 25
              k: 25
            origin: 
              x: -10 
              y: -10 
              z: -10 
            spacing:
              dx: 1
              dy: 1
              dz: 1
          invalid_value: -100.0
      f2: 
        type: "slice"
        params: 
          point: 
            x: 0.0
            y: 0.0
            z: 0.0
          normal: 
            x: 0.0
            y: 0.0
            z: 1.0
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter and sampling past the \
                      mesh dimensions with a large invalid_value.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}
//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_default_values)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a grid of equal size of hexahedron intput");


    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_default_values");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
- 
  action: "add_pipelines"
  pipelines: 
    pl1: 
      f1: 
        type: "sample"
        params: 
          fields: ["braid"]
          uniform_grid:
- 
  action: "add_scenes"
  scenes: 
    s1: 
      plots: 
        p1: 
          type: "pseudocolor"
          field: "braid"
          pipeline: "pl1"
)xyzxyz";
    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter with default values.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}

//-----------------------------------------------------------------------------
TEST(ascent_sample, test_uniform_grid_multiple_fields)
{
    Node n;
    ascent::about(n);
    // only run this test if ascent was built with viskores support
    if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent viskores support disabled, skipping test");
        return;
    }

    //
    // Create an example mesh.
    //
    Node data, verify_info;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              EXAMPLE_MESH_SIDE_DIM,
                                              data);
    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    ASCENT_INFO("Testing sampling a smaller regular grid of hexahedron input");

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_uniform_grid_sample_multiple_fields");

    // remove old images before rendering
    remove_test_image(output_file);

    data["state/cycle"] = 100;
    //
    // Create the actions.
    //

    std::string acts_str = R"xyzxyz(
-
  action: "add_pipelines"
  pipelines:
    pl1:
      f1:
        type: "sample"
        params:
          fields: ["braid","radial"]
          uniform_grid:
            dims:
              i: 10
              j: 10
              k: 10
          invalid_value: -10.0
-
  action: "add_extracts"
  extracts:
    esrc:
      type: relay
      params:
        protocol: hdf5
    eres:
      type: relay
      pipeline: "pl1"
      params:
        protocol: hdf5

-
  action: "add_scenes"
  scenes:
    s1:
      plots:
        p1:
          type: "pseudocolor"
          field: "radial"
          pipeline: "pl1"
)xyzxyz";

    conduit::Node actions;
    actions.parse(acts_str,"yaml");
    actions[1]["extracts/esrc/params/path"] = output_file + "_src";
    actions[1]["extracts/eres/params/path"] = output_file + "_result";
    actions[2]["scenes/s1/renders/r1/image_prefix"] = output_file;
    //actions.print();

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
    EXPECT_TRUE(check_test_image(output_file));
    std::string msg = "An example of using the sample filter with the uniform grid parameter and multiple fields.";
    ASCENT_ACTIONS_DUMP(actions,output_file,msg);
}



//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    if(argc == 2)
    {
        EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
    }

    result = RUN_ALL_TESTS();
    return result;
}
