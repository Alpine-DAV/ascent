//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_mpi_revolve.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <mpi.h>

#include <conduit_blueprint.hpp>

#include <algorithm>
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
}

} // namespace

//-----------------------------------------------------------------------------
TEST(ascent_mpi_revolve, mpi_revolve_rz_r_case_angle_180)
{
  Node n;
  ascent::about(n);
  if(n["runtimes/ascent/viskores/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent viskores support disabled, skipping test");
    return;
  }

  int par_rank;
  int par_size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &par_rank);
  MPI_Comm_size(comm, &par_size);

  ASCENT_INFO("Rank " << par_rank << " of " << par_size << " reporting");

  Node data, verify_info;
  conduit::blueprint::mesh::examples::rz_cylinder("structured", 10, 10, data);
  data["state/domain_id"] = static_cast<uint64>(par_rank);
  data["state/cycle"] = static_cast<uint64>(100);

  // Offset Z per rank to create distinct domains for rendering/compositing.
  const double z_offset = static_cast<double>(par_rank) * 5.0;
  conduit::float64_array z_vals = data["coordsets/coords/values/z"].value();
  for(conduit::index_t i = 0; i < z_vals.number_of_elements(); ++i)
  {
    z_vals[i] += z_offset;
  }

  EXPECT_TRUE(conduit::blueprint::mesh::verify(data, verify_info));

  double rmin = 0.0, rmax = 0.0, zmin = 0.0, zmax = 0.0;
  compute_rz_bounds(data, rmin, rmax, zmin, zmax);

  std::string output_path;
  if(par_rank == 0)
  {
    output_path = prepare_output_dir();
  }
  else
  {
    output_path = output_dir();
  }

  const std::string output_base =
    conduit::utils::join_file_path(output_path, "tout_mpi_revolve_rz_r_case_angle_180_cyl");

  remove_test_image(output_base);

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
  rev_params["angle"] = 180;
  rev_params["steps"] = 8;

  Node &add_scenes = actions.append();
  add_scenes["action"] = "add_scenes";
  Node &scenes = add_scenes["scenes"];

  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "cyl";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/renders/r1/image_prefix"] = output_base;
  scenes["s1/renders/r1/camera/elevation"] = 30;
  scenes["s1/renders/r1/camera/azimuth"] = 90;

  Ascent ascent;
  Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
  ascent_opts["runtime"] = "ascent";
  ascent_opts["exceptions"] = "forward";
  ascent.open(ascent_opts);
  ascent.publish(data);
  ascent.execute(actions);
  ascent.close();

  MPI_Barrier(comm);
  EXPECT_TRUE(check_test_image(output_base, 0.01f));
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
