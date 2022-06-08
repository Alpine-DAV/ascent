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
          it->to_node(dnode);
          // Now, take the dray conduit node and convert to blueprint so
          // we can actually look at it.
          conduit::Node &bnode = n[s.str()];
          dray::BlueprintLowOrder::to_blueprint(dnode, bnode);
      }
      catch(std::exception &e)
      {
          std::cerr << "EXCEPTION:" << e.what() << std::endl;
      }
  }
}

TEST (dray_mesh_threshold, structured)
{

  conduit::Node data;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             data);
  dray::DataSet domain = dray::BlueprintLowOrder::import(data);
  dray::Collection dataset;
  dataset.add_domain(domain);

  dray::MeshThreshold tf;
  tf.set_lower_threshold(0.);
  tf.set_upper_threshold(4.);
  tf.set_field("braid");
  auto tfdataset = tf.execute(dataset);

  // Write the original data.
  conduit::Node inputdata;
  dray_collection_to_blueprint(dataset, inputdata);
  //inputdata.print();
  dray::BlueprintReader::save_blueprint("structured", inputdata);

  // Write the thresholded data to a Blueprint file.
  conduit::Node tfdata;
  dray_collection_to_blueprint(tfdataset, tfdata);
  //tfdata.print();
  //dray::BlueprintReader::save_blueprint("structured_out", tfdata);
}
