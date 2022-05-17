// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"


#include <dray/filters/mesh_boundary.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/rendering/scalar_renderer.hpp>
#include <dray/rendering/slice_plane.hpp>
#include <dray/rendering/surface.hpp>

#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

void setup_camera (dray::Camera &camera)
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

TEST (dray_scalar_renderer, dray_scalars)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "slice_scalars");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_001860.root";

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::Camera camera;
  setup_camera (camera);

  dray::Vec<float, 3> point;
  point[0] = 0.5f;
  point[1] = 0.5f;
  point[2] = 0.5f;

  std::cout<<collection.domain(0).field_info();
  // dray::Vec<float,3> normal;
  std::shared_ptr<dray::SlicePlane> slicer
    = std::make_shared<dray::SlicePlane>(collection);
  //slicer->field("velocity_y");
  slicer->point(point);
  dray::ColorMap color_map("thermal");
  slicer->color_map(color_map);

  dray::ScalarRenderer renderer;
  renderer.set(slicer);
  renderer.field_names(collection.domain(0).fields());
  dray::ScalarBuffer sb = renderer.render(camera);

  conduit::Node mesh;
  sb.to_node(mesh);
  conduit::relay::io::blueprint::save_mesh(mesh, output_file + ".blueprint_root_hdf5");
}

TEST (dray_scalar_renderer, dray_triple_surface)
{
  std::string root_file = std::string(ASCENT_T_DATA_DIR) + "tripple_point/field_dump.cycle_006700.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "triple_scalar");
  remove_test_image (output_file);

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(collection);

  // Camera
  const int c_width = 512;
  const int c_height = 512;
  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.azimuth (-60);
  camera.reset_to_bounds (collection.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("density");

  dray::ScalarRenderer renderer;
  renderer.set(surface);
  renderer.field_names(collection.domain(0).fields());
  dray::ScalarBuffer sb = renderer.render(camera);

  conduit::Node mesh;
  sb.to_node(mesh);
  conduit::relay::io::blueprint::save_mesh(mesh, output_file + ".blueprint_root_hdf5");
}

TEST (dray_scalar_renderer, dray_triple_plane)
{
  std::string root_file = std::string(ASCENT_T_DATA_DIR) + "tripple_point/field_dump.cycle_006700.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "triple_scalar_plane");
  remove_test_image (output_file);

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(collection);

  dray::PlaneDetector det;
  det.m_view = {{0,1,0}};
  det.m_up = {{1,0,0}};
  det.m_center = {{1.5,-9.0,3.5}};
  det.m_x_res = 512;
  det.m_y_res = 512;
  det.m_plane_width = 10.0;
  det.m_plane_height = 10.0;

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("density");

  dray::ScalarRenderer renderer;
  renderer.set(surface);
  renderer.field_names(collection.domain(0).fields());
  dray::ScalarBuffer sb = renderer.render(det);

  conduit::Node mesh;
  sb.to_node(mesh);
  conduit::relay::io::blueprint::save_mesh(mesh, output_file + ".blueprint_root_hdf5");
}


TEST (dray_scalar_renderer, 2comp)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "scalar_render_2comp_vector_test");
  remove_test_image (output_file);
  
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("quads",
                                             100,
                                             100,
                                             0,
                                             data);
  
  conduit::relay::io::blueprint::save_mesh(data, output_file + "_input.blueprint_hdf5","hdf5");


  dray::DataSet domain = dray::BlueprintLowOrder::import(data);

  dray::Collection collection;
  collection.add_domain(domain);

  std::shared_ptr<dray::Surface> surface
     = std::make_shared<dray::Surface>(collection);
  surface->field("vel");


  dray::Camera camera;
  camera.set_width (512);
  camera.set_height (512);
  camera.reset_to_bounds (collection.bounds());

  std::cout<<collection.domain(0).field_info();

  std::vector<std::string> field_names = { "vel" , "braid" };

  dray::ScalarRenderer renderer;
  renderer.set(surface);
  renderer.field_names(field_names);
  dray::ScalarBuffer sb = renderer.render(camera);

  conduit::Node mesh;
  sb.to_node(mesh);
  //mesh.print();
  conduit::relay::io::blueprint::save_mesh(mesh, output_file + "_result.blueprint_hdf5","hdf5");

}


TEST (dray_scalar_renderer, 3comp)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "scalar_render_3comp_vector_test");
  remove_test_image (output_file);
  
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("hexs",
                                             25,
                                             25,
                                             25,
                                             data);
  
  conduit::relay::io::blueprint::save_mesh(data, output_file + "_input.blueprint_hdf5","hdf5");


  dray::DataSet domain = dray::BlueprintLowOrder::import(data);

  dray::Collection collection;
  collection.add_domain(domain);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(collection);

  std::shared_ptr<dray::Surface> surface
     = std::make_shared<dray::Surface>(faces);
  surface->field("vel");


  dray::Camera camera;
  camera.set_width (512);
  camera.set_height (512);
  camera.reset_to_bounds (collection.bounds());

  std::cout<<collection.domain(0).field_info();

  std::vector<std::string> field_names = { "vel" , "braid" };
  
  dray::ScalarRenderer renderer;
  renderer.set(surface);
  renderer.field_names(field_names);
  dray::ScalarBuffer sb = renderer.render(camera);


  conduit::Node mesh;
  sb.to_node(mesh);
  //mesh.print();
  conduit::relay::io::blueprint::save_mesh(mesh, output_file + "_result.blueprint_hdf5","hdf5");

}
