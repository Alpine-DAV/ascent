// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/blueprint_reader.hpp>

#include <dray/filters/vector_component.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/rendering/contour.hpp>
#include <dray/rendering/volume.hpp>

TEST (dray_multi_render, dray_simple)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "multi_render");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_000190.root";

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::VectorComponent vc;
  vc.field("velocity");
  vc.output_name("velocity_y");
  vc.component(1);
  collection = vc.execute(collection);

  dray::Camera camera;
  camera.set_width (512);
  camera.set_height (512);
  camera.reset_to_bounds(collection.bounds());
  camera.azimuth(-40);
  camera.elevate(-40);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::PointLight plight;
  plight.m_pos = { 1.2f, -0.15f, 0.4f };
  plight.m_amb = { 0.3f, 0.3f, 0.3f };
  plight.m_diff = { 0.70f, 0.70f, 0.70f };
  plight.m_spec = { 0.30f, 0.30f, 0.30f };
  plight.m_spec_pow = 90.0;

  dray::AABB<3> bounds = collection.bounds();

  dray::Vec<float, 3> point;
  point[0] = bounds.center()[0];
  point[1] = bounds.center()[1];
  point[2] = bounds.center()[2];

  std::cout<<collection.domain(0).field_info();
  std::shared_ptr<dray::SlicePlane> slicer
    = std::make_shared<dray::SlicePlane>(collection);
  slicer->field("velocity_y");
  slicer->point(point);
  dray::ColorMap color_map("thermal");
  slicer->color_map(color_map);

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(collection);
  volume->field("velocity_y");
  dray::ColorTable tfunc("thermal");
  tfunc.add_alpha(0.1f, 0.f);
  tfunc.add_alpha(1.f, .8f);
  volume->color_map().color_table(tfunc);

  std::shared_ptr<dray::Contour> contour
    = std::make_shared<dray::Contour>(collection);
  contour->field("density");
  contour->iso_field("velocity_y");
  contour->iso_value(0.09);
  contour->color_map(color_map);

  dray::Renderer renderer;
  renderer.add(slicer);
  renderer.add(contour);
  renderer.volume(volume);
  renderer.add_light(plight);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  fb.save_depth("depth");
   // note: dray diff tolerance was 0.2f prior to import
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
}
