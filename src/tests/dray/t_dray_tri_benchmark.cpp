// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/rendering/camera.hpp>
#include <dray/io/obj_reader.hpp>
#include <dray/rendering/triangle_mesh.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/utils/timer.hpp>

#define DRAY_TRIALS 1

TEST (dray_test, dray_test_unit)
{
  std::string file_name = std::string (ASCENT_T_DATA_DIR) + "unit_cube.obj";
  std::cout << "File name " << file_name << "\n";

  std::string output_path = prepare_output_dir ();
  std::string output_base =
  conduit::utils::join_file_path (output_path, "unit_bench");
  remove_test_image (output_base);
  remove_test_image (output_base + "_depth");

  dray::TriangleMesh mesh(file_name);

  dray::Camera camera;
  dray::Vec3f pos = dray::make_vec3f (10, 10, 10);
  dray::Vec3f look_at = dray::make_vec3f (5, 5, 5);
  camera.set_look_at (look_at);
  camera.set_pos (pos);
  camera.reset_to_bounds (mesh.get_bounds ());
  std::cout<<"Bounds "<<mesh.get_bounds()<<"\n";
  dray::Array<dray::Ray> rays;
  dray::Array<dray::RayHit> hits;
  camera.create_rays (rays);
  std::cout << camera.print ();

  dray::Timer timer;
  for (int i = 0; i < DRAY_TRIALS; ++i)
  {
    hits = mesh.intersect(rays);
  }

  float time = timer.elapsed ();
  float ave = time / float (DRAY_TRIALS);
  float ray_size = camera.get_width () * camera.get_height ();
  float rate = (ray_size / ave) / 1e6f;
  std::cout << "Trace rate : " << rate << " (Mray/sec)\n";
  dray::Framebuffer fb(camera.get_width(), camera.get_height());
  mesh.write(rays, hits, fb);
  fb.save(output_base);
  fb.save_depth(output_base + "_depth");

  // allow these to fail, but keep for report
  check_test_image (output_base,dray_baselines_dir());
  check_test_image (output_base + "_depth",dray_baselines_dir());
}

// TEST (dray_test, dray_test_banana)
// {
//   std::string file_name = std::string (ASCENT_T_DATA_DIR) + "banana/banana.obj";
//   std::cout << "File name " << file_name << "\n";
//
//   std::string output_path = prepare_output_dir ();
//   std::string output_file =
//   conduit::utils::join_file_path (output_path, "banana");
//   remove_test_image (output_file);
//
//   dray::TriangleMesh mesh(file_name);
//
//   dray::Camera camera;
//   dray::Vec3f pos = dray::make_vec3f (10, 10, 10);
//   dray::Vec3f look_at = dray::make_vec3f (5, 5, 5);
//   camera.set_look_at (look_at);
//   camera.set_pos (pos);
//   camera.reset_to_bounds (mesh.get_bounds ());
//   camera.set_zoom(1.0);
//   std::cout<<"Bounds "<<mesh.get_bounds()<<"\n";
//   dray::Array<dray::Ray> rays;
//   dray::Array<dray::RayHit> hits;
//   camera.create_rays (rays);
//   std::cout << camera.print ();
//
//   dray::Timer timer;
//   for (int i = 0; i < DRAY_TRIALS; ++i)
//   {
//     hits = mesh.intersect(rays);
//   }
//
//   float time = timer.elapsed ();
//   float ave = time / float (DRAY_TRIALS);
//   float ray_size = camera.get_width () * camera.get_height ();
//   float rate = (ray_size / ave) / 1e6f;
//   std::cout << "Trace rate : " << rate << " (Mray/sec)\n";
//   dray::Framebuffer fb(camera.get_width(), camera.get_height());
//   mesh.shade(rays, hits, fb);
//   fb.save(output_file);
//   fb.save_depth(output_file+"_depth");
//   EXPECT_TRUE (check_test_image (output_file));
// }


// TEST (dray_test, dray_test_conference)
// {
//   std::string file_name = std::string (ASCENT_T_DATA_DIR) + "conference.obj";
//   std::cout << "File name " << file_name << "\n";
//
//   std::string output_path = prepare_output_dir ();
//   std::string output_file =
//   conduit::utils::join_file_path (output_path, "conf_bench");
//   remove_test_image (output_file);
//
//   dray::Array<dray::float32> vertices;
//   dray::Array<dray::int32> indices;
//
//
//   dray::TriangleMesh mesh(file_name);
//   // read_obj (file_name, vertices, indices);
//   //
//   // dray::TriangleMesh mesh (vertices, indices);
//   dray::Camera camera;
//   camera.set_width (1024);
//   camera.set_height (1024);
//
//   dray::Vec3f pos = dray::make_vec3f (30, 19, 5);
//   dray::Vec3f look_at = dray::make_vec3f (0, 0, 0);
//   dray::Vec3f up = dray::make_vec3f (0, 0, 1);
//
//   camera.set_look_at (look_at);
//   camera.set_pos (pos);
//   camera.set_up (up);
//   // camera.reset_to_bounds(mesh.get_bounds());
//   dray::Array<dray::Ray> rays;
//   dray::Array<dray::RayHit> hits;
//   camera.create_rays (rays);
//   std::cout << camera.print ();
//
//   dray::Timer timer;
//   for (int i = 0; i < DRAY_TRIALS; ++i)
//   {
//     mesh.intersect (rays);
//   }
//
//   float time = timer.elapsed ();
//   float ave = time / float (DRAY_TRIALS);
//   float ray_size = camera.get_width () * camera.get_height ();
//   float rate = (ray_size / ave) / 1e6f;
//   std::cout << "Trace rate : " << rate << " (Mray/sec)\n";
//   dray::Framebuffer fb(camera.get_width(), camera.get_height());
//   mesh.shade(rays, hits, fb);
//   fb.save(output_file);
//   fb.save_depth(output_file+"_depth");
//   EXPECT_TRUE (check_test_image (output_file));
// }
