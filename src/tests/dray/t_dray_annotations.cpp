// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/dray.hpp>
#include <dray/color_map.hpp>
#include <dray/rendering/camera.hpp>
#include <dray/rendering/world_annotator.hpp>
#include <sstream>

using namespace dray;

TEST (dray_smoke, dray_world_annotations)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "world_annotations");
  remove_test_image (output_file);

  AABB<3> bounds;
  Vec<float32,3> minb({0.f, 0.f, 0.f});
  const float scale = 0.1f;
  Vec<float32,3> maxb({scale, scale, scale});
  bounds.include(minb);
  bounds.include(maxb);

  const int32 width = 1024;
  const int32 height = 1024;
  Camera camera;
  camera.set_width(width);
  camera.set_height(height);
  camera.reset_to_bounds(bounds);
  //camera.set_zoom(0.8);
  //camera.azimuth(20);
  //camera.elevate(20);

  Array<Ray> rays;
  camera.create_rays(rays);

  Framebuffer fb(camera.get_width(), camera.get_height());
  WorldAnnotator world_annotator(bounds);
  world_annotator.render(fb, rays, camera);

  fb.composite_background();
  fb.save(output_file);
  fb.save_depth(output_file + "_depth");

  EXPECT_TRUE (check_test_image(output_file,dray_baselines_dir()));

}
