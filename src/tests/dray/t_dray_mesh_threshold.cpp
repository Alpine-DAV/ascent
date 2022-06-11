// Copyright 2022 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <conduit_blueprint.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/filters/mesh_threshold.hpp>

int EXAMPLE_MESH_SIDE_DIM = 20;

void
dray_collection_to_blueprint(dray::Collection &c, conduit::Node &n)
{
  int i = 0;
  for(auto it = c.domains().begin();
      it != c.domains().end(); it++, i++)
  {
      std::stringstream s;
      s << "domain" << i << endl;
      conduit::Node dnode;
      try
      {
std::cout << "it->to_node(dnode);" << std::endl;

          it->to_node(dnode);
          // Now, take the dray conduit node and convert to blueprint so
          // we can actually look at it.
          std::string path(s.str());
std::cout << "path=" << path << std::endl;
          conduit::Node &bnode = n[path];
std::cout << "to_blueprint;" << std::endl;
          dray::BlueprintLowOrder::to_blueprint(dnode, bnode);
      }
      catch(std::exception &e)
      {
          std::cerr << "EXCEPTION:" << e.what() << std::endl;
      }
  }
}

void
blueprint_plugin_error_handler(const std::string &msg,
                               const std::string &file,
                               int line)
{
    std::cout << "[ERROR]"
               << "File:"    << file << std::endl
               << "Line:"    << line << std::endl
               << "Message:" << msg  << std::endl;
    while(1); // hang!
}

TEST (dray_mesh_threshold, structured)
{
  conduit::utils::set_error_handler(blueprint_plugin_error_handler);
  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);
  dray::DataSet domain = dray::BlueprintLowOrder::import(data);
  dray::Collection dataset;
  dataset.add_domain(domain);

  // Write the original data.
  conduit::Node inputdata;
  dray_collection_to_blueprint(dataset, inputdata);
  //inputdata.print();
  dray::BlueprintReader::save_blueprint("structured", inputdata);

  // point-centered, any in range.
  dray::MeshThreshold tf;
  tf.set_lower_threshold(-10.);
  tf.set_upper_threshold(0.);
  tf.set_field("braid");
  tf.set_all_in_range(false);
  auto tfdataset = tf.execute(dataset);

  std::cout << "EXECUTED!!!!!!!!!!" << std::endl;

  // Write the thresholded data to a Blueprint file.
  conduit::Node tfdata;
  std::cout << "convert to blueprint" << std::endl;

  dray_collection_to_blueprint(tfdataset, tfdata);
  tfdata.print();
  dray::BlueprintReader::save_blueprint("structured_out", tfdata);

#if 0
  // point-centered, all in range.
  tf.set_all_in_range(true);
  tfdataset = tf.execute(dataset);

  // cell-centered, all in range.
  tf.set_field("radial");
  tf.set_all_in_range(false);
  tfdataset = tf.execute(dataset);


  // cell-centered, all in range.
  tf.set_field("radial");
  tf.set_all_in_range(true);
  tfdataset = tf.execute(dataset);


#endif
}
