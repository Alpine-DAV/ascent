// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/dray.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <mpi.h>

//---------------------------------------------------------------------------//
bool
mfem_enabled()
{
#ifdef DRAY_MFEM_ENABLED
    return true;
#else
    return false;
#endif
}


TEST (dray_volume_render, dray_volume_render_multidom)
{
  if(!mfem_enabled())
  {
    std::cout << "mfem disabled: skipping test that requires high order input " << std::endl;
    return;
  }
  
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "laghos_tg.cycle_000350.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "tg_mpi_volume");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.00f);
  color_table.add_alpha (0.3f, 0.19f);
  color_table.add_alpha (0.4f, 0.21f);
  color_table.add_alpha (1.0f, 0.9f);

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(20);
  camera.elevate(10);

  camera.reset_to_bounds (dataset.bounds());

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(dataset);
  volume->field("density");
  volume->color_map().color_table(color_table);

  dray::Renderer renderer;
  renderer.volume(volume);
  dray::Framebuffer fb = renderer.render(camera);
  if(dray::dray::mpi_rank() == 0)
  {
    fb.composite_background();
    fb.save (output_file);
    // note: dray diff tolerance was 0.2f prior to import
    EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
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
