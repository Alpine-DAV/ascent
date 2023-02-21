// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/volume.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/filters/volume_balance.hpp>

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

  int trials = 5;
  // parse any custon info out of config
  if (config.m_config.has_path ("trials"))
  {
    trials = config.m_config["trials"].to_int32 ();
  }

  dray::ColorTable color_table("Spectral");

  if(!config.has_color_table())
  {
    color_table.add_alpha (0.f, 0.4f);
    color_table.add_alpha (0.1f, 0.0f);
    color_table.add_alpha (0.2f, 0.0f);
    color_table.add_alpha (0.3f, 0.2f);
    color_table.add_alpha (0.4f, 0.2f);
    color_table.add_alpha (0.5f, 0.5f);
    color_table.add_alpha (0.6f, 0.5f);
    color_table.add_alpha (0.7f, 0.4f);
    color_table.add_alpha (0.8f, 0.3f);
    color_table.add_alpha (0.9f, 0.2f);
    color_table.add_alpha (1.0f, 0.8f);
  }
  else
  {
    config.load_color_table();
    color_table = config.m_color_table;
  }

  int samples = 100;
  if(config.m_config.has_path("samples"))
  {
    samples = config.m_config["samples"].to_int32();
  }

  if(config.m_config.has_path("load_balance"))
  {
    bool load_balance = config.m_config["load_balance"].as_string() == "true";
    float32 factor = 0.9;
    bool use_prefix = true;
    if(config.m_config.has_path("factor"))
    {
      factor = config.m_config["factor"].to_float32();
    }
    if(config.m_config.has_path("use_prefix"))
    {
      use_prefix = config.m_config["use_prefix"].as_string() == "true";
    }


    dray::VolumeBalance balancer;
    balancer.prefix_balancing(use_prefix);
    balancer.piece_factor(factor);
    dray::Collection res = balancer.execute(config.m_collection,config.m_camera, samples);
    config.m_collection = res;
  }

  std::shared_ptr<dray::Volume> volume
    = std::make_shared<dray::Volume>(config.m_collection);
  volume->field(config.m_field);
  volume->use_lighting(true);
  volume->samples(samples);


  if(config.m_config.has_path("alpha_scale"))
  {
    float scale = config.m_config["alpha_scale"].to_float32();
    volume->color_map().alpha_scale(scale);
  }

  volume->color_map().color_table(color_table);
  volume->color_map().scalar_range(config.range());
  volume->color_map().log_scale(config.log_scale());

  dray::Array<dray::VolumePartial> partials;
  int rank = dray::dray::mpi_rank();

  dray::Renderer renderer;
  renderer.volume(volume);

  dray::Framebuffer fb;
  for (int i = 0; i < trials; ++i)
  {
    fb = renderer.render(config.m_camera);
  }

  if(dray::dray::mpi_rank() == 0)
  {
    fb.composite_background();
    fb.save("volume");
  }

  dray::stats::StatStore::write_ray_stats ("out_furnace_volume_rendering",
                                           config.m_camera.get_width (),
                                           config.m_camera.get_height ());

  finalize_furnace();
}
