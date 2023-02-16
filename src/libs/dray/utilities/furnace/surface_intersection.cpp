// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/png_encoder.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <iostream>
#include <random>


int main (int argc, char *argv[])
{
  init_furnace();

  std::string config_file = "";

  if (argc != 2)
  {
    std::cout << "Missing configure file name\n";
    exit (1);
  }

  config_file = argv[1];

  Config config (config_file);
  config.load_data ();
  config.load_camera ();
  config.load_field ();

  dray::MeshBoundary boundary;
  dray::Collection faces = boundary.execute(config.m_collection);

  int trials = 5;
  // parse any custon info out of config
  if (config.m_config.has_path ("trials"))
  {
    trials = config.m_config["trials"].to_int32 ();
  }

  std::shared_ptr<dray::Surface> surface
    = std::make_shared<dray::Surface>(faces);
  surface->field(config.m_field);

  dray::Framebuffer framebuffer;

  dray::Renderer renderer;
  renderer.add(surface);

  for (int i = 0; i < trials; ++i)
  {
    framebuffer = renderer.render(config.m_camera);
  }

  if(dray::dray::mpi_rank() == 0)
  {
    framebuffer.composite_background();
    framebuffer.save ("surface_intersection");
    framebuffer.save_depth("surface_intersection_depth");
  }

  dray::stats::StatStore::write_ray_stats ("out_furnace_surface_intersection",
                                           config.m_camera.get_width (),
                                           config.m_camera.get_height ());

  finalize_furnace();
}
