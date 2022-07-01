// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/utils/appstats.hpp>

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

  int num_points = 1000;
  int trials = 5;
  // parse any custon info out of config
  if (config.m_config.has_path ("trials"))
  {
    trials = config.m_config["trials"].to_int32 ();
  }
  if (config.m_config.has_path ("points"))
  {
    num_points = config.m_config["points"].to_int32 ();
  }

  dray::AABB<3> bounds = config.m_collection.bounds();

  dray::Array<dray::Vec<dray::Float, 3>> points;
  points.resize (num_points);

  // random but deterministic
  std::linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647> rgen{ 0 };
  std::uniform_real_distribution<dray::Float> dist_x{ bounds.m_ranges[0].min (),
                                                      bounds.m_ranges[0].max () };

  std::uniform_real_distribution<dray::Float> dist_y{ bounds.m_ranges[1].min (),
                                                      bounds.m_ranges[1].max () };

  std::uniform_real_distribution<dray::Float> dist_z{ bounds.m_ranges[2].min (),
                                                      bounds.m_ranges[2].max () };

  dray::Vec<dray::Float, 3> *points_ptr = points.get_host_ptr ();

  for (int i = 0; i < num_points; ++i)
  {
    dray::Vec<dray::Float, 3> point;
    point[0] = dist_x (rgen);
    point[1] = dist_y (rgen);
    point[2] = dist_z (rgen);
    points_ptr[i] = point;
  }


  dray::Array<dray::Location> locations;
  const int domains = config.m_collection.local_size();
  for (int i = 0; i < trials; ++i)
  {
    for(int d = 0; d < domains; ++d)
    {
      locations = config.m_collection.domain(d).mesh()->locate (points);
    }
  }

  dray::stats::StatStore::write_point_stats ("locate_stats");
  finalize_furnace();
}
