// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/blueprint_reader.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/filters/vector_component.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/utils/appstats.hpp>

dray::PointLight default_light(dray::Camera &camera)
{
  dray::Vec<float32,3> look_at = camera.get_look_at();
  dray::Vec<float32,3> pos = camera.get_pos();
  dray::Vec<float32,3> up = camera.get_up();
  up.normalize();
  dray::Vec<float32,3> look = look_at - pos;
  dray::float32 mag = look.magnitude();
  dray::Vec<float32,3> right = cross (look, up);
  right.normalize();

  dray::Vec<float32, 3> miner_up = cross (right, look);
  miner_up.normalize();
  dray::Vec<float32, 3> light_pos = pos + .1f * mag * miner_up;
  dray::PointLight light;
  light.m_pos = light_pos;
  return light;
}


void setup_camera_slice(dray::Camera &camera)
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

void setup_camera_three_slice(dray::Camera &camera)
{
  camera.set_width (512);
  camera.set_height (512);

  dray::Vec<dray::float32, 3> pos;
  pos[0] = -3.0f;
  pos[1] =  2.0f;
  pos[2] = -3.0f;
  camera.set_up (dray::make_vec3f (0, 0, 1));
  camera.set_pos (pos);
  camera.set_look_at (dray::make_vec3f (0.5, 0.5, 0.5));
}


TEST (dray_slice, dray_slice)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "slice");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_001860.root";

  dray::Collection collection = dray::BlueprintReader::load (root_file);
  for(auto name : collection.domain(0).fields()) std::cout<<"Field "<<name<<"\n";

  dray::VectorComponent vc;
  vc.field("velocity");
  vc.output_name("velocity_y");
  vc.component(1);
  collection = vc.execute(collection);

  dray::Camera camera;
  setup_camera_slice(camera);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());

  dray::PointLight plight;
  plight.m_pos = { 1.2f, -0.15f, 0.4f };
  plight.m_amb = { 1.0f, 1.0f, 1.f };
  plight.m_diff = { 0.0f, 0.0f, 0.0f };
  plight.m_spec = { 0.0f, 0.0f, 0.0f };
  plight.m_spec_pow = 90.0;

  dray::Vec<float, 3> point;
  point[0] = 0.5f;
  point[1] = 0.5f;
  point[2] = 0.5f;

  std::cout<<collection.domain(0).field_info();
  // dray::Vec<float,3> normal;
  std::shared_ptr<dray::SlicePlane> slicer
    = std::make_shared<dray::SlicePlane>(collection);
  slicer->field("velocity_y");
  slicer->point(point);
  dray::ColorMap color_map("thermal");
  slicer->color_map(color_map);

  dray::Renderer renderer;
  renderer.add(slicer);
  renderer.add_light(plight);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  //dray::stats::StatStore::write_point_stats ("locate_stats");
  // note: dray diff tolerance was 0.2f prior to import
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
}


TEST (dray_slice, dray_three_slice)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "three_slice");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_001860.root";

  dray::Collection collection = dray::BlueprintReader::load (root_file);
  for(auto name : collection.domain(0).fields()) std::cout<<"Field "<<name<<"\n";

  dray::Camera camera;
  setup_camera_three_slice(camera);

  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  dray::Framebuffer framebuffer (camera.get_width (), camera.get_height ());


  dray::PointLight plight = default_light(camera);

  dray::Vec<float, 3> point;
  point[0] = 0.5f;
  point[1] = 0.5f;
  point[2] = 0.5f;
  
  dray::Vec<float,3> x_normal({1.f, 0.f, 0.f});
  dray::Vec<float,3> y_normal({0.f, 1.f, 0.f});
  dray::Vec<float,3> z_normal({0.f, 0.f, 1.f});
  
  dray::ColorMap color_map("cool2warm");
  
  std::shared_ptr<dray::SlicePlane> slicer_x
    = std::make_shared<dray::SlicePlane>(collection);

  std::shared_ptr<dray::SlicePlane> slicer_y
    = std::make_shared<dray::SlicePlane>(collection);

  std::shared_ptr<dray::SlicePlane> slicer_z
    = std::make_shared<dray::SlicePlane>(collection);

  slicer_x->field("density");
  slicer_y->field("density");
  slicer_z->field("density");

  slicer_x->color_map(color_map);
  slicer_y->color_map(color_map);
  slicer_z->color_map(color_map);

  slicer_x->point(point);
  slicer_x->normal(x_normal);

  slicer_y->point(point);
  slicer_y->normal(y_normal);

  slicer_z->point(point);
  slicer_z->normal(z_normal);

  std::cout<<collection.domain(0).field_info();

  dray::Renderer renderer;
  renderer.add(slicer_x);
  renderer.add(slicer_y);
  renderer.add(slicer_z);
  renderer.add_light(plight);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save (output_file);
  //dray::stats::StatStore::write_point_stats ("locate_stats");
  // note: dray diff tolerance was 0.2f prior to import
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir(),0.05));
}
