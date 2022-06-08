// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/filters/mesh_boundary.hpp>
#include <dray/rendering/surface.hpp>
#include <dray/rendering/scalar_renderer.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/png_encoder.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <conduit_relay.hpp>
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

  dray::ScalarRenderer renderer;
  renderer.set(surface);
  renderer.field_names(config.m_collection.domain(0).fields());
  dray::ScalarBuffer sb = renderer.render(config.m_camera);

  //for (int i = 0; i < trials; ++i)
  //{
  //  framebuffer = renderer.render(config.m_camera);
  //}

  if(dray::dray::mpi_rank() == 0)
  {
    conduit::Node mesh;
    sb.to_node(mesh);
    conduit::relay::io::blueprint::save_mesh(mesh, "scalars.blueprint_root_hdf5");
  }

  dray::stats::StatStore::write_ray_stats (config.m_camera.get_width (),
                                           config.m_camera.get_height ());

  finalize_furnace();
}
