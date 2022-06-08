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

int EXAMPLE_MESH_SIDE_DIM = 5;

void
dray_collection_to_conduit(dray::Collection &c, conduit::Node &n)
{
  int i = 0;
  for(auto it = c.domains().begin();
      it != c.domains().end(); it++, i++)
  {
      std::stringstream s;
      s << "domain" << i << endl;
      conduit::Node &dnode = n[s.str()];
      it->to_node(dnode);
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

  // Print the original data.
  conduit::Node inputdata;
  dray_collection_to_conduit(dataset, inputdata);
  inputdata.print();

  // Write the thresholded data to a Blueprint file.
  conduit::Node tfdata;
  dray_collection_to_conduit(tfdataset, tfdata);
  tfdata.print();
  //dray::BlueprintReader::save_blueprint("structured.bp", tfdata);
}
