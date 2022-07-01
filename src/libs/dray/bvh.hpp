// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BVH_HPP
#define DRAY_BVH_HPP

#include <dray/aabb.hpp>
#include <dray/array.hpp>

namespace dray
{

struct BVH
{
  Array<Vec<float32, 4>> m_inner_nodes;
  Array<int32> m_leaf_nodes;
  AABB<> m_bounds; // total bounds of primitives
  Array<int32> m_aabb_ids;
  // multiple leaf nodes can point to the same
  // original primitive. m_aabb_ids point to the
  // index of the aabb given to construct

  // BVH layout
  // root is at the beginning of the array
  // each bvh node is 4 x Vec<float32,4>s for a total of 16 values
  // [0-5] left child aabb (xmin,ymin,zmin)-(xmax,ymax,zmax)
  // [6-11] right child aabb (xmin,ymin,zmin)-(xmax,ymax,zmax)
  // [12] index of left child
  // [13] index of right child
  // Child index notes: these values are ints stored in floating
  //  point values. Use memcpy or reinterpret_cast to get the actual
  //  integer back.
  //  positive indices: inner node.
  //  negative values: child node. To ge the index of the child
  //    real index = -index - 1
  // [14-15] un-used
};

} // namespace dray
#endif
