// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/rendering/renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>


TEST (dray_volume_render, dray_volume_render_simple)
{
  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "impeller_p2_000000.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "impeller_vr");
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

  camera.reset_to_bounds (dataset.bounds());

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(dataset);
  volume->field("diffusion");
  volume->color_map().color_table(color_table);

  dray::Renderer renderer;
  renderer.volume(volume);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
}

TEST (dray_volume_render, dray_volume_render_triple)
{
  std::string root_file = std::string(ASCENT_T_DATA_DIR) + "tripple_point/field_dump.cycle_006700.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "triple_vr");
  remove_test_image (output_file);

  dray::Collection dataset = dray::BlueprintReader::load (root_file);

  dray::ColorTable color_table ("Spectral");
  color_table.add_alpha (0.f, 0.00f);
  color_table.add_alpha (0.1f, 0.20f);
  color_table.add_alpha (0.4f, 0.9f);
  color_table.add_alpha (0.9f, 0.61f);
  color_table.add_alpha (1.0f, 0.9f);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth (-60);
  camera.reset_to_bounds (dataset.bounds());

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);

  dray::Framebuffer framebuffer (c_width, c_height);
  framebuffer.clear ();

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(dataset);
  volume->field("density");
  volume->use_lighting(false);
  volume->color_map().color_table(color_table);

  dray::Renderer renderer;
  renderer.volume(volume);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
}
