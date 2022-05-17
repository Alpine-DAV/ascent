// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/dray.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <mpi.h>

TEST (dray_volume_render, dray_volume_render_multidom)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "laghos_tg.cycle_000350.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "tg_mpi_faces");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(dataset);

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(20);
  camera.elevate(10);

  camera.reset_to_bounds (dataset.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("density");
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);

  if(dray::dray::mpi_rank() == 0)
  {
    fb.composite_background();
    fb.save (output_file);
    EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
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
