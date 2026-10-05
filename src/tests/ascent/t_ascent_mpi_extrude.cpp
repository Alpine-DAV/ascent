//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_mpi_extrude.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <mpi.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace conduit;
using namespace ascent;

//-----------------------------------------------------------------------------
TEST(ascent_mpi_extrude, mpi_linear_extrude_rz_structured)
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
    conduit::utils::join_file_path(output_path, "tout_mpi_extrude_rz_structured");

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

  Node &add_scenes = actions.append();
  add_scenes["action"] = "add_scenes";
  Node &scenes = add_scenes["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "cyl";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/image_prefix"] = output_base;

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

