// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/rendering/camera.hpp>
#include <dray/rendering/contour.hpp>
#include <dray/filters/vector_component.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/io/blueprint_reader.hpp>

TEST (dray_isosurface, simple)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "isosurface_simple");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_000190.root";

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::VectorComponent vc;
  vc.field("velocity");
  vc.output_name("velocity_x");
  vc.component(0);
  collection = vc.execute(collection);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(-40);

  camera.reset_to_bounds (collection.bounds());

  dray::ColorTable color_table ("ColdAndHot");
  // dray::Vec<float,3> normal;

  const float isoval = 0.09;

  std::shared_ptr<dray::Contour> contour
    = std::make_shared<dray::Contour>(collection);
  contour->field("density");
  contour->iso_field("velocity_x");
  contour->iso_value(isoval);
  contour->color_map().color_table(color_table);;

  dray::Renderer renderer;
  renderer.add(contour);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
}

TEST (dray_isosurface, complex)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "isosurface");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_001860.root";

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::VectorComponent vc;
  vc.field("velocity");
  vc.output_name("velocity_x");
  vc.component(0);
  collection = vc.execute(collection);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth(-40);

  camera.reset_to_bounds (collection.bounds());
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::ColorTable color_table ("ColdAndHot");
  // dray::Vec<float,3> normal;

  const float isoval = 0.09;

  std::shared_ptr<dray::Contour> contour
    = std::make_shared<dray::Contour>(collection);
  contour->field("density");
  contour->iso_field("velocity_x");
  contour->iso_value(isoval);
  contour->color_map().color_table(color_table);;

  dray::Renderer renderer;
  renderer.add(contour);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
}
