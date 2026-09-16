//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_extrude.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
TEST(ascent_extrude, test_linear_extrude_rz_structured)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  Node data, verify_info;
  conduit::blueprint::mesh::examples::rz_cylinder("structured", 10, 10, data);
  data["state/cycle"] = 100;
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  std::string output_path = prepare_output_dir();
  std::string output_base = conduit::utils::join_file_path(output_path, "tout_extrude_structured");
  std::string output_extract_root = output_base + "_hdf5";

  conduit::utils::remove_directory(output_extract_root);
  remove_test_file(output_extract_root + ".cycle_000100.root");
  remove_test_image(output_base);

  Node actions;

  Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  Node &pipelines = add_pipelines["pipelines"];

  pipelines["pl1/f1/type"] = "extrude";
  Node &ext_params = pipelines["pl1/f1/params"];
  ext_params["vector/r"] = 0.0;
  ext_params["vector/z"] = 5.0;
  ext_params["steps"] = 8;

  Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  Node &extracts = add_extracts["extracts"];
  extracts["e1/type"] = "relay";
  extracts["e1/pipeline"] = "pl1";
  extracts["e1/params/path"] = output_extract_root;
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

  Node &add_scenes = actions.append();
  add_scenes["action"] = "add_scenes";
  Node &scenes = add_scenes["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "cyl";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/renders/r1/image_prefix"] = output_base;

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_base, 0.01f));
  EXPECT_TRUE(check_test_file(output_extract_root + ".cycle_000100.root"));

  std::stringstream ss;
  ss << "An example of linearly extruding a dataset 8 steps along a vector.";
  ASCENT_ACTIONS_DUMP(actions,output_base,ss.str());
}

//-----------------------------------------------------------------------------
TEST(ascent_extrude, test_linear_extrude_rz_unstructured)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  Node data, verify_info;
  conduit::blueprint::mesh::examples::rz_cylinder("unstructured", 10, 10, data);
  data["state/cycle"] = 100;
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  std::string output_path = prepare_output_dir();
  std::string output_base = conduit::utils::join_file_path(output_path, "tout_extrude_unstructured");
  std::string output_extract_root = output_base + "_hdf5";

  conduit::utils::remove_directory(output_extract_root);
  remove_test_file(output_extract_root + ".cycle_000100.root");
  remove_test_image(output_base);

  Node actions;

  Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  Node &pipelines = add_pipelines["pipelines"];

  pipelines["pl1/f1/type"] = "extrude";
  Node &ext_params = pipelines["pl1/f1/params"];
  ext_params["vector/r"] = 0.0;
  ext_params["vector/z"] = 5.0;
  ext_params["steps"] = 8;

  Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  Node &extracts = add_extracts["extracts"];
  extracts["e1/type"] = "relay";
  extracts["e1/pipeline"] = "pl1";
  extracts["e1/params/path"] = output_extract_root;
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";

  Node &add_scenes = actions.append();
  add_scenes["action"] = "add_scenes";
  Node &scenes = add_scenes["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "cyl";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/renders/r1/image_prefix"] = output_base;

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_base, 0.01f));
  EXPECT_TRUE(check_test_file(output_extract_root + ".cycle_000100.root"));

  std::stringstream ss;
  ss << "An example of linearly extruding a dataset 8 steps along a vector.";
  ASCENT_ACTIONS_DUMP(actions,output_base,ss.str());
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  result = RUN_ALL_TESTS();
  return result;
}
