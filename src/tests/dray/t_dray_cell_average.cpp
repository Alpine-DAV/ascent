// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <conduit/conduit.hpp>
#include <dray/filters/cell_average.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/dray_node_to_dataset.hpp>

#include <iostream>

const int EXAMPLE_MESH_SIDE_DIM = 3;

//---------------------------------------------------------------------------//
bool
mfem_enabled()
{
#ifdef DRAY_MFEM_ENABLED
  return true;
#else
  return false;
#endif
}

TEST(dray_cell_average, smoke)
{
  std::cout << "Hello, world!" << std::endl;
  dray::CellAverage filter;
  (void)filter;
}

TEST(dray_cell_average, braid_structured)
{
  // Load the example mesh
  conduit::Node mesh;
  conduit::blueprint::mesh::examples::braid("structured",
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             EXAMPLE_MESH_SIDE_DIM,
                                             mesh);

  // Convert it to a dray collection
  dray::DataSet domain = dray::BlueprintLowOrder::import(mesh);
  std::cout << domain.field_info() << std::endl;
  dray::Collection collection;
  collection.add_domain(domain);
  std::cout << collection.domain(0).field_info() << std::endl;

  // Fields:
  //   braid  - point centered
  //   radial - cell centered
  //   vel    - point centered (vector)
  dray::CellAverage filter;
  filter.set_field("braid");
  auto result = filter.execute(collection);
  std::cout << result.domain(0).field_info() << std::endl;
}
