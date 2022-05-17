// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/dray.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/rendering/scalar_renderer.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/rendering/surface.hpp>

#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

#include <mpi.h>

void setup_camera (dray::Camera &camera)
{
  camera.set_width (512);
  camera.set_height (512);

  dray::Vec<dray::float32, 3> pos;
  pos[0] = .5f;
  pos[1] = -1.5f;
  pos[2] = .5f;
  camera.set_up (dray::make_vec3f (0, 0, 1));
  camera.set_pos (pos);
  camera.set_look_at (dray::make_vec3f (0.5, 0.5, 0.5));
}

TEST (dray_scalar_renderer, dray_triple_surface)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "laghos_tg.cycle_000350.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "scalar_mpi");
  remove_test_image (output_file);

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(collection);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth (-60);
  camera.reset_to_bounds (collection.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("density");

  dray::ScalarRenderer renderer;
  renderer.set(surface);
  renderer.field_names(collection.domain(0).fields());
  dray::ScalarBuffer sb = renderer.render(camera);


  if(dray::dray::mpi_rank() == 0)
  {
    conduit::Node mesh;
    sb.to_node(mesh);
    conduit::relay::io::blueprint::save_mesh(mesh, output_file + ".blueprint_root_hdf5");
  }
}

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
