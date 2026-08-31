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

namespace
{

std::string
find_root_file(const std::string &output_extract_root)
{
  if(conduit::utils::is_file(output_extract_root + ".root"))
  {
    return output_extract_root + ".root";
  }
  if(conduit::utils::is_file(output_extract_root + ".cycle_000100.root"))
  {
    return output_extract_root + ".cycle_000100.root";
  }
  if(conduit::utils::is_file(output_extract_root + ".cycle_000000.root"))
  {
    return output_extract_root + ".cycle_000000.root";
  }
  return "";
}

void
verify_extrude_output(const std::string &output_extract_root,
                      const std::string &expected_shape,
                      const std::string &check_axis)
{
  Node res, res_verify;
  const std::string root_file = find_root_file(output_extract_root);
  EXPECT_FALSE(root_file.empty());

  conduit::relay::io::load(root_file + ":mesh", "hdf5", res);
  EXPECT_TRUE(conduit::blueprint::mesh::verify(res, res_verify));

  std::vector<std::string> topo_names = res["topologies"].child_names();
  EXPECT_TRUE(topo_names.size() > 0);
  const std::string topo_name = topo_names[0];

  if(res["topologies/" + topo_name + "/elements"].has_path("shape"))
  {
    EXPECT_EQ(res["topologies/" + topo_name + "/elements/shape"].as_string(), expected_shape);
  }

  const index_t out_points = res["coordsets/coords/values/x"].dtype().number_of_elements();
  EXPECT_TRUE(out_points > 0);
  const double *vals = nullptr;
  if(check_axis == "x")
  {
    vals = res["coordsets/coords/values/x"].as_double_ptr();
  }
  else if(check_axis == "y")
  {
    vals = res["coordsets/coords/values/y"].as_double_ptr();
  }
  else if(check_axis == "z")
  {
    vals = res["coordsets/coords/values/z"].as_double_ptr();
  }
  else
  {
    FAIL() << "Unsupported axis '" << check_axis << "'";
  }

  double vmin = vals[0], vmax = vals[0];
  for(index_t i = 1; i < out_points; ++i)
  {
    vmin = std::min(vmin, vals[i]);
    vmax = std::max(vmax, vals[i]);
  }
  EXPECT_TRUE((vmax - vmin) > 0.0);
}

} // namespace

//-----------------------------------------------------------------------------
TEST(ascent_extrude, test_linear_extrude_tris)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  Node data, verify_info;
  conduit::blueprint::mesh::examples::braid("tris", 20, 20, 0, data);
  data["state/cycle"] = 100;
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  std::string output_path = prepare_output_dir();
  std::string output_base = conduit::utils::join_file_path(output_path, "tout_extrude_tris");
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
  ext_params["vector/x"] = 0.0;
  ext_params["vector/y"] = 0.0;
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
  scenes["s1/plots/p1/type"] = "mesh";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/renders/r1/image_prefix"] = output_base;
  scenes["s1/renders/r1/camera/look_at"] = {0.0, 0.0, 0.0};
  scenes["s1/renders/r1/camera/position"] = {30.0, 30.0, 30.0};
  scenes["s1/renders/r1/camera/up"] = {0.0, 0.0, 1.0};
  scenes["s1/renders/r1/camera/zoom"] = 0.8;

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_base, 0.01f));

  verify_extrude_output(output_extract_root, "wedge", "z");
}

//-----------------------------------------------------------------------------
TEST(ascent_extrude, test_linear_extrude_quads)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  Node data, verify_info;
  conduit::blueprint::mesh::examples::braid("quads", 20, 20, 0, data);
  data["state/cycle"] = 100;
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  std::string output_path = prepare_output_dir();
  std::string output_base = conduit::utils::join_file_path(output_path, "tout_extrude_quads");
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
  ext_params["vector/x"] = 0.0;
  ext_params["vector/y"] = 0.0;
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
  scenes["s1/plots/p1/type"] = "mesh";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/renders/r1/image_prefix"] = output_base;
  scenes["s1/renders/r1/camera/look_at"] = {0.0, 0.0, 0.0};
  scenes["s1/renders/r1/camera/position"] = {30.0, 30.0, 30.0};
  scenes["s1/renders/r1/camera/up"] = {0.0, 0.0, 1.0};
  scenes["s1/renders/r1/camera/zoom"] = 0.8;

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  
  EXPECT_TRUE(check_test_image(output_base, 0.01f));

  verify_extrude_output(output_extract_root, "hex", "z");
}

//-----------------------------------------------------------------------------
TEST(ascent_extrude, test_linear_extrude_rz_cylinder)
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
  std::string output_base = conduit::utils::join_file_path(output_path, "tout_extrude_rz");
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
  // rz_cylinder is embedded in 3D with a constant axis; extrude along Y to ensure non-zero y extent.
  ext_params["vector/x"] = 0.0;
  ext_params["vector/y"] = 5.0;
  ext_params["vector/z"] = 0.0;
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
  scenes["s1/renders/r1/camera/look_at"] = {0.0, 0.0, 0.0};
  scenes["s1/renders/r1/camera/position"] = {20.0, 20.0, 20.0};
  scenes["s1/renders/r1/camera/up"] = {0.0, 1.0, 0.0};

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_base, 0.01f));

  verify_extrude_output(output_extract_root, "hex", "y");
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  result = RUN_ALL_TESTS();
  return result;
}
