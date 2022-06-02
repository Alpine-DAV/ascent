// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "tconfig.h"
#include "t_utils.hpp"

#include <dray/camera.hpp>
#include <dray/io/obj_reader.hpp>
#include <dray/triangle_mesh.hpp>
#include <dray/utils/ray_utils.hpp>

TEST (dray_test, dray_test_unit)
{
  std::string file_name = std::string (ASCENT_T_DATA_DIR) + "unit_cube.obj";
  std::cout << "File name " << file_name << "\n";

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "unit_cube_depth");
  remove_test_image (output_file);

  dray::Array<dray::float32> vertices;
  dray::Array<dray::int32> indices;

  read_obj (file_name, vertices, indices);

  dray::TriangleMesh mesh (vertices, indices);
  dray::Camera camera;
  dray::Vec3f pos = dray::make_vec3f (10, 10, 10);
  dray::Vec3f look_at = dray::make_vec3f (5, 5, 5);
  camera.set_look_at (look_at);
  camera.set_pos (pos);
  camera.reset_to_bounds (mesh.get_bounds ());
  dray::Array<dray::Ray> rays;
  camera.create_rays (rays);
  std::cout << camera.print ();
  mesh.intersect (rays);

  dray::save_depth (rays, camera.get_width (), camera.get_height (), output_file);

  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
}

// TEST(dray_test, dray_test_conference)
//{
//  std::string file_name = std::string(ASCENT_T_DATA_DIR) + "conference.obj";
//  std::cout<<"File name "<<file_name<<"\n";
//
//  dray::Array<dray::float32> vertices;
//  dray::Array<dray::int32> indices;
//
//  read_obj(file_name, vertices, indices);
//
//  dray::TriangleMesh mesh(vertices, indices);
//  dray::Camera camera;
//
//  dray::Vec3f pos = dray::make_vec3f(30,19,5);
//  dray::Vec3f look_at = dray::make_vec3f(0,0,0);
//  dray::Vec3f up = dray::make_vec3f(0,0,1);
//
//  camera.set_look_at(look_at);
//  camera.set_pos(pos);
//  camera.set_up(up);
//  //camera.reset_to_bounds(mesh.get_bounds());
//  dray::ray32 rays;
//  camera.create_rays(rays);
//  std::cout<<camera.print();
//  mesh.intersect(rays);
//
//  dray::save_depth(rays, camera.get_width(), camera.get_height());
//
//}
