// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/io/blueprint_reader.hpp>
#include <dray/dray_node_to_dataset.hpp>

#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>

#include <dray/math.hpp>

#include <fstream>
#include <stdlib.h>

TEST (dray_dataset_to_node, dray_node_round_trip)
{
  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green_2d.cycle_000050.root";
  std::string output_path = prepare_output_dir ();

  dray::Collection collection = dray::BlueprintReader::load (root_file);

  dray::DataSet domain = collection.domain(0);;
  conduit::Node n_dataset;
  domain.to_node(n_dataset);

  dray::DataSet out_dataset = dray::to_dataset(n_dataset);

  std::string output_file =
  conduit::utils::join_file_path (output_path, "round_trip");
  // Camera
  const int c_width  = 1024;
  const int c_height = 1024;

  dray::Collection col;
  col.add_domain(out_dataset);

  dray::Camera camera;
  camera.set_width (c_width);
  camera.set_height (c_height);
  camera.reset_to_bounds (col.bounds());

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(col);
  surface->field("density");
  surface->draw_mesh (true);
  surface->line_thickness(.1);

  dray::Renderer renderer;
  renderer.add(surface);
  dray::Framebuffer fb = renderer.render(camera);

  fb.save(output_file);

}
