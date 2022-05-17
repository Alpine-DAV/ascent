// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <conduit_blueprint.hpp>

#include <dray/io/blueprint_reader.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/utils/appstats.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

int EXAMPLE_MESH_SIDE_DIM = 20;

void render_2d(conduit::Node &data, std::string name)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, name);
  remove_test_image (output_file);

  dray::DataSet domain = dray::BlueprintLowOrder::import(data);

  dray::Collection dataset;
  dataset.add_domain(domain);

  dray::ColorTable color_table ("cool2warm");

  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (dataset.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(dataset);
  surface->field("braid");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
  fb.save_depth (output_file + "_depth");
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}


void render_3d(conduit::Node &data, std::string name)
{

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, name);
  remove_test_image (output_file);

  dray::DataSet domain = dray::BlueprintLowOrder::import(data);

  dray::Collection dataset;
  dataset.add_domain(domain);

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(dataset);

  dray::ColorTable color_table ("cool2warm");

  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.elevate(10);
  camera.azimuth(40);
  camera.reset_to_bounds (dataset.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field("braid");
  surface->color_map().color_table(color_table);
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);
  fb.composite_background();

  fb.save(output_file);
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
  fb.save_depth (output_file + "_depth");
  dray::stats::StatStore::write_ray_stats (c_width, c_height);
}

TEST (dray_low_order, dray_uniform_quads)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             0,
                                             data);
  render_2d(data, "uniform_quads");
}

TEST (dray_low_order, dray_uniform_hexs)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("uniform",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  render_3d(data, "uniform_hexs");
}

TEST (dray_low_order, dray_explicit_hexs)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("hexs",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  render_3d(data, "explicit_hexs");
}
TEST (dray_low_order, dray_explicit_tets)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("tets",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  render_3d(data, "explicit_tets");
}

TEST (dray_low_order, dray_explicit_tris)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("tris",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  render_2d(data, "explicit_tris");
}

TEST (dray_low_order, dray_structured_quads)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             1,
                                             data);

  render_2d(data, "structured_quads");
}


TEST (dray_low_order, dray_structured_hexs)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);

  render_3d(data, "structured_hexs");
}
