// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/reflect.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_reflect, dray_reflect_2d)
{
  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green_2d.cycle_000050.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "tg_2d_reflect");
  remove_test_image (output_file);

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::Vec<float,3> point = {0.f, 0.f, 0.f};
  dray::Vec<float,3> normal = {0.f, 1.f, 0.f};

  dray::Reflect reflector;
  reflector.plane(point, normal);
  dray::Collection reflected = reflector.execute(collection);

  dray::AABB<3> bounds;
  bounds.include(collection.bounds());
  bounds.include(reflected.bounds());

  dray::ColorTable color_table ("Spectral");

  // Camera
  const int c_width = 512;
  const int c_height = 512;

  std::string field_name = "density";

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (bounds);

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(collection);
  surface->field(field_name);
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  std::shared_ptr<dray::Surface> surface2
    = std::make_shared<dray::Surface>(reflected);
  surface2->field(field_name);
  surface2->color_map().color_table(color_table);
  surface2->draw_mesh (true);
  surface2->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  renderer.add(surface2);
  dray::Framebuffer fb = renderer.render(camera);

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
  fb.save_depth (output_file + "_depth");
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}
