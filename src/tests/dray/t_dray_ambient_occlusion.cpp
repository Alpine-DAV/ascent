// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_config.hpp"

#include <dray/ambient_occlusion.hpp>
#include <dray/camera.hpp>
#include <dray/io/obj_reader.hpp>
#include <dray/triangle_mesh.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/utils/timer.hpp>

#include <fstream>
#include <iostream>

#define DRAY_TRIALS 20

// TEST(dray_test, dray_test_unit)
void cancel_test_cube ()
{
  // Input the data from disk.
  std::string file_name = std::string (ASCENT_T_DATA_DIR) + "unit_cube.obj";
  std::cout << "File name " << file_name << "\n";

  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj (file_name, vertices, indices);

  // Build the scene/camera.
  dray::TriangleMesh mesh (vertices, indices);
  dray::Camera camera;
  dray::Vec3f pos = dray::make_vec3f (.9, .9, .9);
  dray::Vec3f look_at = dray::make_vec3f (.5, .5, .5);
  camera.set_look_at (look_at);
  camera.set_pos (pos);

  // camera.reset_to_bounds(mesh.get_bounds());

  camera.set_width (500);
  camera.set_height (500);

  dray::Array<dray::Ray> primary_rays;
  camera.create_rays (primary_rays); // Must be after setting camera width, height.
  std::cout << camera.print ();

  dray::AABB<> mesh_bounds = mesh.get_bounds ();

  std::cerr << "The bounds are " << mesh_bounds << std::endl;

  dray::float32 mesh_scaling =
  max (max (mesh_bounds.m_ranges[0].length (), mesh_bounds.m_ranges[1].length ()),
       mesh_bounds.m_ranges[2].length ());

  mesh.intersect (primary_rays);

  dray::save_depth (primary_rays, camera.get_width (), camera.get_height ());

  // Generate occlusion rays.
  dray::int32 occ_samples = 50;

  dray::Array<dray::IntersectionContext> intersection_ctx =
  mesh.get_intersection_context (primary_rays);

  dray::Array<dray::int32> compact_indexing_array;

  dray::Array<dray::Ray> occ_rays =
  dray::AmbientOcclusion::gen_occlusion (intersection_ctx, occ_samples, .000000001f,
                                         0.03 * mesh_scaling, compact_indexing_array);

  const dray::int32 *compact_indexing = compact_indexing_array.get_host_ptr_const ();

  mesh.intersect (occ_rays);

  /// // Write out OBJ for some lines.
  /// const dray::Vec3f *orig_ptr = occ_rays.m_orig.get_host_ptr_const();
  /// const dray::Vec3f *dir_ptr = occ_rays.m_dir.get_host_ptr_const();
  /// const dray::float32 *dist_ptr = occ_rays.m_dist.get_host_ptr_const();
  /// const dray::float32 *far_ptr = occ_rays.m_far.get_host_ptr_const();
  /// const dray::int32 *pid_ptr = occ_rays.m_pixel_id.get_host_ptr_const();
  /// std::ofstream obj_output;
  /// obj_output.open("occ_rays.obj");
  /// dray::int32 test_num_bundles = 2;
  /// dray::int32 test_bundle_idxs[] = {compact_indexing[f5], compact_indexing[f6]}; //f5 and f6 used to be specific pixel ids.
  /// for (dray::int32 test_idx = 0; test_idx < test_num_bundles; test_idx++)
  /// {
  ///   dray::int32 test_offset = test_bundle_idxs[test_idx] * occ_samples;

  ///   std::cerr << "OBJ loop: pixel_id == " << pid_ptr[test_offset] << std::endl;

  ///   for (dray::int32 i = 0; i < occ_samples; i++)
  ///   {
  ///     dray::int32 occ_ray_idx = i + test_bundle_idxs[test_idx] * occ_samples;

  ///     // Get ray origin and endpoint.
  ///     dray::Vec3f orig = orig_ptr[occ_ray_idx];
  ///     //dray::Vec3f tip = orig_ptr[occ_ray_idx] + dir_ptr[occ_ray_idx] * dist_ptr[occ_ray_idx]; //Using hit point.
  ///     dray::Vec3f tip = orig_ptr[occ_ray_idx] + dir_ptr[occ_ray_idx] * far_ptr[occ_ray_idx]; //Using ray "far".
  ///
  ///     // Output two vertices and then connect them.
  ///     obj_output << "v " << orig[0] << " " << orig[1] << " " << orig[2] << std::endl;
  ///     obj_output << "v " << tip[0] << " " << tip[1] << " " << tip[2] << std::endl;
  ///     obj_output << "l " << 2*i+1 + test_idx*occ_samples << " " << 2*i+2 + test_idx*occ_samples << std::endl;
  ///   }
  /// }
  /// obj_output.close();

  // dray::save_hitrate(occ_rays, occ_samples, camera.get_width(), camera.get_height());
}

#if 0
//TEST(dray_test, dray_test_conference)
void cancel_test2()
{
  // Input the data from disk.
  std::string file_name = std::string(ASCENT_T_DATA_DIR) + "conference.obj";
  std::cout<<"File name "<<file_name<<"\n";

  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  // Build the scene/camera.
  dray::TriangleMesh mesh(vertices, indices);
  dray::Camera camera;
  camera.set_width(2048);
  camera.set_height(2048);

  dray::AABB<> mesh_bounds = mesh.get_bounds();
  dray::float32 mesh_scaling =
      max(max(mesh_bounds.m_x.length(),
              mesh_bounds.m_y.length()),
              mesh_bounds.m_z.length());

  dray::Vec3f pos = dray::make_vec3f(30,19,5);
  dray::Vec3f look_at = dray::make_vec3f(0,0,0);
  dray::Vec3f up = dray::make_vec3f(0,0,1);

  camera.set_look_at(look_at);
  camera.set_pos(pos);
  camera.set_up(up);
  //camera.reset_to_bounds(mesh.get_bounds());

  dray::Ray primary_rays;
  camera.create_rays(primary_rays);
  std::cout<<camera.print();

  mesh.intersect(primary_rays);

  // Generate occlusion rays.
  dray::int32 occ_samples = 50;

  dray::IntersectionContext<dray::float32> intersection_ctx = mesh.get_intersection_context(primary_rays);

  //dray::Ray occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 300.0f);
  dray::Ray occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 0.03 * mesh_scaling);

  mesh.intersect(occ_rays);

  dray::save_hitrate(occ_rays, occ_samples, camera.get_width(), camera.get_height());
}


TEST(dray_test, dray_test_city)
//void cancel_test3()
{
  // Input the data from disk.
  std::string file_name = std::string(ASCENT_T_DATA_DIR) + "city_triangulated.obj";
  std::cout<<"File name "<<file_name<<"\n";

  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj(file_name, vertices, indices);

  // Build the scene/camera.
  dray::TriangleMesh mesh(vertices, indices);
  dray::Camera camera;
  camera.set_width(1024);
  camera.set_height(1024);

  dray::AABB<> mesh_bounds = mesh.get_bounds();
  dray::float32 mesh_scaling =
      max(max(mesh_bounds.m_x.length(),
              mesh_bounds.m_y.length()),
              mesh_bounds.m_z.length());

  //dray::Vec3f pos = dray::make_vec3f(0.0f, 0.65f, -0.75f);
  dray::Vec3f pos = dray::make_vec3f(0.156f, 0.455f, -0.42f);
  dray::Vec3f look_at = dray::make_vec3f(0.52f, 0.0f, 0.35f);
  dray::Vec3f up = dray::make_vec3f(0,1,0);

  camera.set_look_at(look_at);
  camera.set_pos(pos);
  camera.set_up(up);
  camera.reset_to_bounds(mesh.get_bounds());

  dray::Ray primary_rays;
  camera.create_rays(primary_rays);
  std::cout<<camera.print();

  mesh.intersect(primary_rays);

  // Generate occlusion rays.
  dray::int32 occ_samples = 100;

  dray::IntersectionContext<dray::float32> intersection_ctx = mesh.get_intersection_context(primary_rays);

  dray::Ray occ_rays = dray::AmbientOcclusion<dray::float32>::gen_occlusion(intersection_ctx, occ_samples, .000000001f, 0.03 * mesh_scaling);

  mesh.intersect(occ_rays);

  dray::save_hitrate(occ_rays, occ_samples, camera.get_width(), camera.get_height());
}
#endif
