// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LINEAR_BVH_BUILDER_HPP
#define DRAY_LINEAR_BVH_BUILDER_HPP

#include <dray/aabb.hpp>
#include <dray/array.hpp>
#include <dray/bvh.hpp>

namespace dray
{

class LinearBVHBuilder
{

  public:
  BVH construct (Array<AABB<>> aabbs);
  BVH construct (Array<AABB<>> aabbs, Array<int32> primimitive_ids);
};

AABB<> reduce (const Array<AABB<>> &aabbs);

} // namespace dray
#endif
