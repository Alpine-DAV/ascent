//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_revolve.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

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
load_output_mesh(const std::string &output_extract_root,
                 Node &mesh,
                 std::string &topo_name)
{
  Node verify_info;
  const std::string root_file = find_root_file(output_extract_root);
  EXPECT_FALSE(root_file.empty());
  if(root_file.empty())
  {
    return;
  }

  conduit::relay::io::load(root_file + ":mesh", "hdf5", mesh);
  EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, verify_info));

  std::vector<std::string> topo_names = mesh["topologies"].child_names();
  EXPECT_TRUE(!topo_names.empty());
  if(topo_names.empty())
  {
    return;
  }

  topo_name = topo_names[0];
}

void
verify_revolve_mesh_checks(const std::string &output_extract_root,
                           const int steps,
                           const bool periodic,
                           const std::string &expected_shape)
{
  Node res;
  std::string topo_name;
  load_output_mesh(output_extract_root, res, topo_name);
  if(topo_name.empty())
  {
    return;
  }

  const std::string topo_path = "topologies/" + topo_name;
  const std::string conn_path = topo_path + "/elements/connectivity";

  EXPECT_TRUE(res[conn_path].dtype().number_of_elements() > 0);

  if(!expected_shape.empty() && res[topo_path + "/elements"].has_path("shape"))
  {
    EXPECT_EQ(res[topo_path + "/elements/shape"].as_string(), expected_shape);
  }

  const index_t out_points = res["coordsets/coords/values/x"].dtype().number_of_elements();
  EXPECT_TRUE(out_points > 0);
  if(out_points <= 0)
  {
    return;
  }

  const double *x = res["coordsets/coords/values/x"].as_double_ptr();
  const double *y = res["coordsets/coords/values/y"].as_double_ptr();
  const double *z = res["coordsets/coords/values/z"].as_double_ptr();

  double xmin = x[0], xmax = x[0];
  double ymin = y[0], ymax = y[0];
  double zmin = z[0], zmax = z[0];

  for(index_t i = 1; i < out_points; ++i)
  {
    xmin = std::min(xmin, x[i]);
    xmax = std::max(xmax, x[i]);
    ymin = std::min(ymin, y[i]);
    ymax = std::max(ymax, y[i]);
    zmin = std::min(zmin, z[i]);
    zmax = std::max(zmax, z[i]);
  }

  EXPECT_TRUE((xmax - xmin) > 0.0);
  EXPECT_TRUE((ymax - ymin) > 0.0);
  EXPECT_TRUE((zmax - zmin) > 0.0);

  const index_t planes = periodic ? steps : (steps + 1);
  EXPECT_TRUE(planes > 0);
  EXPECT_TRUE(out_points % planes == 0);
  if(planes <= 0 || (out_points % planes) != 0)
  {
    return;
  }

  const index_t points_per_plane = out_points / planes;
  EXPECT_TRUE(points_per_plane > 0);
  if(points_per_plane <= 0)
  {
    return;
  }

  bool any_moved = false;
  const index_t sample = std::min<index_t>(points_per_plane, 32);
  for(index_t i = 0; i < sample; ++i)
  {
    const index_t j = i + points_per_plane;
    if(j >= out_points)
    {
      break;
    }
    const double dx = std::abs(x[j] - x[i]);
    const double dy = std::abs(y[j] - y[i]);
    const double dz = std::abs(z[j] - z[i]);
    if((dx + dy + dz) > 1e-6)
    {
      any_moved = true;
      break;
    }
  }
  EXPECT_TRUE(any_moved);

  const index_t conn_len = res[conn_path].dtype().number_of_elements();
  EXPECT_TRUE(conn_len > 0);
  if(conn_len <= 0)
  {
    return;
  }

  const conduit::DataType conn_dt = res[conn_path].dtype();
  index_t max_id = 0;
  if(conn_dt.is_int32())
  {
    const int32 *conn = res[conn_path].as_int32_ptr();
    max_id = static_cast<index_t>(conn[0]);
    for(index_t i = 1; i < conn_len; ++i)
    {
      max_id = std::max(max_id, static_cast<index_t>(conn[i]));
    }
  }
  else if(conn_dt.is_int64())
  {
    const int64 *conn = res[conn_path].as_int64_ptr();
    max_id = static_cast<index_t>(conn[0]);
    for(index_t i = 1; i < conn_len; ++i)
    {
      max_id = std::max(max_id, static_cast<index_t>(conn[i]));
    }
  }
  else if(conn_dt.is_uint32())
  {
    const uint32 *conn = res[conn_path].as_uint32_ptr();
    max_id = static_cast<index_t>(conn[0]);
    for(index_t i = 1; i < conn_len; ++i)
    {
      max_id = std::max(max_id, static_cast<index_t>(conn[i]));
    }
  }
  else if(conn_dt.is_uint64())
  {
    const uint64 *conn = res[conn_path].as_uint64_ptr();
    max_id = static_cast<index_t>(conn[0]);
    for(index_t i = 1; i < conn_len; ++i)
    {
      max_id = std::max(max_id, static_cast<index_t>(conn[i]));
    }
  }
  else
  {
    FAIL() << "Unexpected connectivity dtype: " << conn_dt.name();
    return;
  }

  EXPECT_TRUE(max_id >= points_per_plane);
}

void
verify_revolve_z_extent(const std::string &output_extract_root,
                        const std::string &expected_shape)
{
  Node res;
  std::string topo_name;
  load_output_mesh(output_extract_root, res, topo_name);
  if(topo_name.empty())
  {
    return;
  }

  const std::string topo_path = "topologies/" + topo_name;
  if(!expected_shape.empty() && res[topo_path + "/elements"].has_path("shape"))
  {
    EXPECT_EQ(res[topo_path + "/elements/shape"].as_string(), expected_shape);
  }

  const index_t out_points = res["coordsets/coords/values/x"].dtype().number_of_elements();
  EXPECT_TRUE(out_points > 0);
  if(out_points <= 0)
  {
    return;
  }

  const double *z = res["coordsets/coords/values/z"].as_double_ptr();

  double zmin = z[0], zmax = z[0];
  for(index_t i = 1; i < out_points; ++i)
  {
    zmin = std::min(zmin, z[i]);
    zmax = std::max(zmax, z[i]);
  }
  EXPECT_TRUE((zmax - zmin) > 0.0);
}

void
run_revolve_contour_lines_case(const double angle, const bool periodic)
{
  Node data, verify_info;

  conduit::blueprint::mesh::examples::braid("hexs", 20, 20, 20, data);
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  const int steps = 8;

  std::ostringstream mesh, pseudo;
  mesh << "tout_revolve_angle_" << static_cast<int>(angle) << "_mesh";
  pseudo << "tout_revolve_angle_" << static_cast<int>(angle) << "_braid";
  std::string output_path = prepare_output_dir();
  std::string output_base = conduit::utils::join_file_path(output_path, mesh.str());
  std::string output_base_pseudo = conduit::utils::join_file_path(output_path, pseudo.str());
  std::string output_extract_root = output_base + "_hdf5";

  conduit::utils::remove_directory(output_extract_root);
  remove_test_file(output_extract_root + ".cycle_000100.root");
  remove_test_image(output_base);
  remove_test_image(output_base_pseudo);

  Node actions;

  Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  Node &pipelines = add_pipelines["pipelines"];

  pipelines["pl1/f1/type"] = "slice";
  Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["point/x"] = 0.0;
  slice_params["point/y"] = 0.0;
  slice_params["point/z"] = 0.0;
  slice_params["normal/x"] = 0.0;
  slice_params["normal/y"] = 0.0;
  slice_params["normal/z"] = 1.0;

  pipelines["pl1/f2/type"] = "contour";
  Node &contour_params = pipelines["pl1/f2/params"];
  contour_params["field"] = "braid";
  contour_params["iso_values"] = 0.0;

  pipelines["pl1/f3/type"] = "revolve";
  Node &rev_params = pipelines["pl1/f3/params"];
  rev_params["point/x"] = 0.0;
  rev_params["point/y"] = 0.0;
  rev_params["point/z"] = 0.0;
  rev_params["axis/x"] = 0.0;
  rev_params["axis/y"] = 1.0;
  rev_params["axis/z"] = 0.0;
  rev_params["angle"] = angle;
  rev_params["steps"] = steps;
  rev_params["periodic"] = periodic ? "true" : "false";

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
  scenes["s1/renders/r1/camera/position"] = {20.0, 20.0, 20.0};
  scenes["s1/renders/r1/camera/up"] = {0.0, 1.0, 0.0};
  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/field"] = "braid";
  scenes["s2/plots/p1/pipeline"] = "pl1";
  scenes["s2/renders/r1/image_prefix"] = output_base_pseudo;
  scenes["s2/renders/r1/camera/look_at"] = {0.0, 0.0, 0.0};
  scenes["s2/renders/r1/camera/position"] = {20.0, 20.0, 20.0};
  scenes["s2/renders/r1/camera/up"] = {0.0, 1.0, 0.0};

  // print our full actions tree
  std::cout << actions.to_yaml() << std::endl;

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_base, 0.01f));
  EXPECT_TRUE(check_test_image(output_base_pseudo, 0.01f));

  verify_revolve_mesh_checks(output_extract_root, steps, periodic, "quad");
  std::stringstream ss;
  ss << "An example of revolving (rotationally extruding) a dataset " << angle << " degrees over " << steps << " steps.";
  ASCENT_ACTIONS_DUMP(actions,output_base,ss.str());
}

void
run_revolve_slice_surface_case(const double angle, const bool periodic)
{
  Node data, verify_info;

  conduit::blueprint::mesh::examples::braid("hexs", 20, 20, 20, data);
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  const int steps = 8;

  std::ostringstream mesh, pseudo;
  mesh << "tout_revolve_surface_angle_" << static_cast<int>(angle) << "_mesh";
  pseudo << "tout_revolve_surface_angle_" << static_cast<int>(angle) << "_braid";
  std::string output_path = prepare_output_dir();
  std::string output_base = conduit::utils::join_file_path(output_path, mesh.str());
  std::string output_base_pseudo = conduit::utils::join_file_path(output_path, pseudo.str());
  std::string output_extract_root = output_base + "_hdf5";

  conduit::utils::remove_directory(output_extract_root);
  remove_test_file(output_extract_root + ".cycle_000100.root");
  remove_test_image(output_base);
  remove_test_image(output_base_pseudo);

  Node actions;

  Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  Node &pipelines = add_pipelines["pipelines"];

  pipelines["pl1/f1/type"] = "slice";
  Node &slice_params = pipelines["pl1/f1/params"];
  slice_params["point/x"] = 0.0;
  slice_params["point/y"] = 0.0;
  slice_params["point/z"] = 0.0;
  slice_params["normal/x"] = 0.0;
  slice_params["normal/y"] = 0.0;
  slice_params["normal/z"] = 1.0;

  pipelines["pl1/f2/type"] = "revolve";
  Node &rev_params = pipelines["pl1/f2/params"];
  rev_params["point/x"] = 0.0;
  rev_params["point/y"] = 0.0;
  rev_params["point/z"] = 0.0;
  rev_params["axis/x"] = 0.0;
  rev_params["axis/y"] = 1.0;
  rev_params["axis/z"] = 0.0;
  rev_params["angle"] = angle;
  rev_params["steps"] = steps;
  rev_params["periodic"] = periodic ? "true" : "false";

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
  scenes["s1/renders/r1/camera/position"] = {20.0, 20.0, 20.0};
  scenes["s1/renders/r1/camera/up"] = {0.0, 1.0, 0.0};
  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/field"] = "braid";
  scenes["s2/plots/p1/pipeline"] = "pl1";
  scenes["s2/renders/r1/image_prefix"] = output_base_pseudo;
  scenes["s2/renders/r1/camera/look_at"] = {0.0, 0.0, 0.0};
  scenes["s2/renders/r1/camera/position"] = {20.0, 20.0, 20.0};
  scenes["s2/renders/r1/camera/up"] = {0.0, 1.0, 0.0};

  // print our full actions tree
  std::cout << actions.to_yaml() << std::endl;

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_base, 0.01f));
  EXPECT_TRUE(check_test_image(output_base_pseudo, 0.01f));

  verify_revolve_z_extent(output_extract_root, "wedge");
  std::stringstream ss;
  ss << "An example of revolving (rotationally extruding) a dataset " << angle << " degrees over " << steps << " steps.";
  ASCENT_ACTIONS_DUMP(actions,output_base,ss.str());
}

void
run_revolve_rz_case(const double angle, const bool periodic)
{
  Node data, verify_info;

  conduit::blueprint::mesh::examples::rz_cylinder("structured", 10, 10, data);
  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  const int steps = 8;

  std::ostringstream mesh, pseudo;
  mesh << "tout_revolve_rz_angle_" << static_cast<int>(angle) << "_mesh";
  pseudo << "tout_revolve_rz_angle_" << static_cast<int>(angle) << "_cyl";
  std::string output_path = prepare_output_dir();
  std::string output_base = conduit::utils::join_file_path(output_path, mesh.str());
  std::string output_base_pseudo = conduit::utils::join_file_path(output_path, pseudo.str());
  std::string output_extract_root = output_base + "_hdf5";

  conduit::utils::remove_directory(output_extract_root);
  remove_test_file(output_extract_root + ".cycle_000100.root");
  remove_test_image(output_base);
  remove_test_image(output_base_pseudo);

  Node actions;

  Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  Node &pipelines = add_pipelines["pipelines"];

  // rz_cylinder produces quad cells; revolve currently expects triangles.
  pipelines["pl1/f1/type"] = "triangulate";
  pipelines["pl1/f2/type"] = "revolve";
  Node &rev_params = pipelines["pl1/f2/params"];
  rev_params["point/x"] = 0.0;
  rev_params["point/y"] = 0.0;
  rev_params["point/z"] = 0.0;
  rev_params["axis/x"] = 0.0;
  rev_params["axis/y"] = 1.0;
  rev_params["axis/z"] = 0.0;
  rev_params["angle"] = angle;
  rev_params["steps"] = steps;
  rev_params["periodic"] = periodic ? "true" : "false";

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
  scenes["s1/renders/r1/camera/position"] = {20.0, 20.0, 20.0};
  scenes["s1/renders/r1/camera/up"] = {0.0, 1.0, 0.0};
  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/field"] = "cyl";
  scenes["s2/plots/p1/pipeline"] = "pl1";
  scenes["s2/renders/r1/image_prefix"] = output_base_pseudo;
  scenes["s2/renders/r1/camera/look_at"] = {0.0, 0.0, 0.0};
  scenes["s2/renders/r1/camera/position"] = {20.0, 20.0, 20.0};
  scenes["s2/renders/r1/camera/up"] = {0.0, 1.0, 0.0};

  // print our full actions tree
  std::cout << actions.to_yaml() << std::endl;

  Ascent ascent;
  ascent.open();
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  EXPECT_TRUE(check_test_image(output_base, 0.01f));
  EXPECT_TRUE(check_test_image(output_base_pseudo, 0.01f));

  verify_revolve_mesh_checks(output_extract_root, steps, periodic, "wedge");
  std::stringstream ss;
  ss << "An example of revolving (rotationally extruding) a dataset " << angle << " degrees over " << steps << " steps.";
  ASCENT_ACTIONS_DUMP(actions,output_base,ss.str());
}

} // namespace

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_contour_lines_angle_90)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_contour_lines_case(90.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_contour_lines_angle_180)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_contour_lines_case(180.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_contour_lines_angle_270)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_contour_lines_case(270.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_contour_lines_angle_360)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_contour_lines_case(360.0, true);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_slice_surface_angle_90)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_slice_surface_case(90.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_slice_surface_angle_180)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_slice_surface_case(180.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_slice_surface_angle_270)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_slice_surface_case(270.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_slice_surface_angle_360)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_slice_surface_case(360.0, true);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_angle_90)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_rz_case(90.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_angle_180)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_rz_case(180.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_angle_270)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_rz_case(270.0, false);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_angle_360)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  run_revolve_rz_case(360.0, true);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    result = RUN_ALL_TESTS();
    return result;
}
