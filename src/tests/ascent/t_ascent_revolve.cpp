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

void
compute_rz_bounds(const Node &mesh,
                  double &rmin,
                  double &rmax,
                  double &zmin,
                  double &zmax)
{
  EXPECT_TRUE(mesh.has_path("coordsets"));
  if(!mesh.has_path("coordsets"))
  {
    rmin = rmax = zmin = zmax = 0.0;
    return;
  }

  const std::vector<std::string> cs_names = mesh["coordsets"].child_names();
  EXPECT_TRUE(!cs_names.empty());
  if(cs_names.empty())
  {
    rmin = rmax = zmin = zmax = 0.0;
    return;
  }

  const Node &cs = mesh["coordsets/" + cs_names[0]];
  Node cs_explicit;
  conduit::blueprint::mesh::coordset::to_explicit(cs, cs_explicit);

  EXPECT_TRUE(cs_explicit.has_path("values/r"));
  EXPECT_TRUE(cs_explicit.has_path("values/z"));
  if(!cs_explicit.has_path("values/r") || !cs_explicit.has_path("values/z"))
  {
    rmin = rmax = zmin = zmax = 0.0;
    return;
  }

  Node r_node, z_node;
  cs_explicit["values/r"].to_float64_array(r_node);
  cs_explicit["values/z"].to_float64_array(z_node);

  const index_t n = r_node.dtype().number_of_elements();
  EXPECT_TRUE(n > 0);
  EXPECT_EQ(n, z_node.dtype().number_of_elements());
  if(n <= 0 || z_node.dtype().number_of_elements() != n)
  {
    rmin = rmax = zmin = zmax = 0.0;
    return;
  }

  const double *r = r_node.as_float64_ptr();
  const double *z = z_node.as_float64_ptr();

  rmin = rmax = r[0];
  zmin = zmax = z[0];
  for(index_t i = 1; i < n; ++i)
  {
    rmin = std::min(rmin, r[i]);
    rmax = std::max(rmax, r[i]);
    zmin = std::min(zmin, z[i]);
    zmax = std::max(zmax, z[i]);
  }
  //std::cerr << "rmin: " << rmin << " rmax: " << rmax << " zmin: " << zmin << " zmax: " << zmax << std::endl;
}


void
run_revolve_rz_r_case(const double angle, const bool periodic)
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

  double rmin = 0.0, rmax = 0.0, zmin = 0.0, zmax = 0.0;
  compute_rz_bounds(data, rmin, rmax, zmin, zmax);

  const int steps = 8;
  const int angle_int = static_cast<int>(angle);

  std::ostringstream mesh, pseudo;
  mesh << "tout_revolve_rz_r_case_angle_" << angle_int << "_mesh";
  pseudo << "tout_revolve_rz_r_case_angle_" << angle_int << "_cyl";

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
  // Rotate about the Z axis (in RZ), at the lower (r,z) corner to avoid self-intersection.
  rev_params["point/r"] = rmin;
  rev_params["point/z"] = zmin;
  rev_params["axis/r"] = 1.0;
  rev_params["axis/z"] = 0.0;
  rev_params["angle"] = angle_int;
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
  scenes["s1/renders/r1/camera/elevation"] = 30;
  scenes["s1/renders/r1/camera/azimuth"] = 90;

  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/field"] = "cyl";
  scenes["s2/plots/p1/pipeline"] = "pl1";
  scenes["s2/renders/r1/image_prefix"] = output_base_pseudo;
  scenes["s2/renders/r1/camera/elevation"] = 30;
  scenes["s2/renders/r1/camera/azimuth"] = 90;

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();
  
  //too much to ask for the mesh to be exact?
  //EXPECT_TRUE(check_test_image(output_base, 0.01f));
  EXPECT_TRUE(check_test_image(output_base_pseudo, 0.01f));
  EXPECT_TRUE(check_test_file(output_extract_root) +".cycle_000100.root");

  std::stringstream ss;
  ss << "An example of revolving (rotationally extruding) a dataset " << angle_int
     << " degrees over " << steps << " steps.";
  ASCENT_ACTIONS_DUMP(actions, output_base, ss.str());
}

void
run_revolve_rz_z_case(const double angle, const bool periodic)
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

  double rmin = 0.0, rmax = 0.0, zmin = 0.0, zmax = 0.0;
  compute_rz_bounds(data, rmin, rmax, zmin, zmax);

  const int steps = 8;
  const int angle_int = static_cast<int>(angle);

  std::ostringstream mesh, pseudo;
  mesh << "tout_revolve_rz_z_case_angle_" << angle_int << "_mesh";
  pseudo << "tout_revolve_rz_z_case_angle_" << angle_int << "_cyl";

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
  // Rotate about the Z axis (in RZ), at the lower (r,z) corner to avoid self-intersection.
  rev_params["point/r"] = rmin;
  rev_params["point/z"] = zmin;
  rev_params["axis/r"] = 0.0;
  rev_params["axis/z"] = 1.0;
  rev_params["angle"] = angle_int;
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
  scenes["s1/renders/r1/camera/elevation"] = 30;
  scenes["s1/renders/r1/camera/azimuth"] = 90;

  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/field"] = "cyl";
  scenes["s2/plots/p1/pipeline"] = "pl1";
  scenes["s2/renders/r1/image_prefix"] = output_base_pseudo;
  scenes["s2/renders/r1/camera/elevation"] = 30;
  scenes["s2/renders/r1/camera/azimuth"] = 90;

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["runtime/type"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();
  
  //too much to ask for the mesh to be exact?
  //EXPECT_TRUE(check_test_image(output_base, 0.01f));
  EXPECT_TRUE(check_test_image(output_base_pseudo, 0.01f));
  EXPECT_TRUE(check_test_file(output_extract_root) +".cycle_000100.root");

  std::stringstream ss;
  ss << "An example of revolving (rotationally extruding) a dataset " << angle_int
     << " degrees over " << steps << " steps.";
  ASCENT_ACTIONS_DUMP(actions, output_base, ss.str());
}

} // namespace

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_r_case_angle_90)
{
  const bool periodic = false;
  run_revolve_rz_r_case(90.0, periodic);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_r_case_angle_180)
{
  const bool periodic = false;
  run_revolve_rz_r_case(180.0, periodic);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_r_case_angle_270)
{
  const bool periodic = false;
  run_revolve_rz_r_case(270.0, periodic);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_r_case_angle_360)
{
  const bool periodic = true;
  run_revolve_rz_r_case(360.0, periodic);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_z_case_angle_90)
{
  const bool periodic = false;
  run_revolve_rz_z_case(90.0, periodic);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_z_case_angle_180)
{
  const bool periodic = false;
  run_revolve_rz_z_case(180.0, periodic);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_z_case_angle_270)
{
  const bool periodic = false;
  run_revolve_rz_z_case(270.0, periodic);
}

//-----------------------------------------------------------------------------
TEST(ascent_revolve, test_revolve_rz_z_case_angle_360)
{
  const bool periodic = true;
  run_revolve_rz_z_case(360.0, periodic);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    result = RUN_ALL_TESTS();
    return result;
}
