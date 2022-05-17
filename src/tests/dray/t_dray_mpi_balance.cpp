// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/dray.hpp>
#include <dray/filters/volume_balance.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <mpi.h>

TEST (dray_redistribute, redistribute)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "laghos_tg.cycle_000350.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
    conduit::utils::join_file_path (output_path, "balanced");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  int rank = dray::dray::mpi_rank();
  int size = dray::dray::mpi_size();

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(20);
  camera.elevate(10);

  camera.reset_to_bounds (dataset.bounds());

  int32 samples = 100;

  dray::VolumeBalance balancer;
  dray::Collection res = balancer.execute(dataset, camera, samples);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.00f);
  color_table.add_alpha (0.3f, 0.19f);
  color_table.add_alpha (0.4f, 0.21f);
  color_table.add_alpha (1.0f, 0.9f);

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(res);
  volume->field("density");
  volume->samples(samples);
  volume->color_map().color_table(color_table);

  dray::Renderer renderer;
  renderer.volume(volume);
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
