// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/dray.hpp>
#include <dray/color_map.hpp>
#include <dray/rendering/billboard.hpp>
#include <dray/rendering/camera.hpp>
#include <sstream>
using namespace dray;

TEST (dray_smoke, dray_billboard)
{
  std::string output_path = prepare_output_dir ();
  std::string output_base =
  conduit::utils::join_file_path (output_path, "billboard_test");

  std::vector<std::string> texts;
  std::vector<Vec<float32,3>> positions;
  std::vector<Vec<float32,2>> offsets;
  std::vector<float32> text_sizes;
  texts.push_back("Bananas");
  Vec<float32,3> pos({0.f, 0.f, 0.f});
  Vec<float32,2> offset({0.f, 0.f});
  positions.push_back(pos);
  offsets.push_back(offset);
  text_sizes.push_back(20);


  Billboard billboard(texts, positions, offsets, text_sizes);
  AABB<3> bounds = billboard.bounds();

  const int c_width  = 512;
  const int c_height = 512;

  Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (bounds);
  for(int i = 0; i < 20; ++i)
  {
    camera.elevate(5);
    camera.azimuth(5);
    Array<Ray> rays;
    camera.create_rays(rays);
    billboard.camera(camera);

    Array<RayHit> hits = billboard.intersect(rays);

    Framebuffer fb(c_width, c_height);
    billboard.shade(rays, hits, fb);
    std::stringstream ss;
    ss <<  output_base <<"_text_billboard_"<<std::setfill('0') << std::setw(4)<<i;
    std::string output_file = ss.str();
    remove_test_image (output_file);
    fb.save(output_file);
    fb.save_depth(output_file + "_depth");
    EXPECT_TRUE (check_test_image(output_file,dray_baselines_dir()));
  }
}
