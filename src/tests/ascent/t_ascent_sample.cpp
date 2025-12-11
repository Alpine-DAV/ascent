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
    // only run this test if ascent was built with vtkm support
    if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
    {
        ASCENT_INFO("Ascent vtkm support disabled, skipping test");
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


