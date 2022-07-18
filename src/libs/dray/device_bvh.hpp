// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_BVH_HPP
#define DRAY_DEVICE_BVH_HPP

#include <dray/bvh.hpp>

namespace dray
{

struct DeviceBVH
{
  const Vec<float32, 4> *m_inner_nodes;
  const int32 *m_leaf_nodes;
  AABB<> m_bounds;
  const int32 *m_aabb_ids;

  DeviceBVH () = delete;
  DeviceBVH (const BVH &bvh)
  : m_inner_nodes (bvh.m_inner_nodes.get_device_ptr_const ()),
    m_leaf_nodes (bvh.m_leaf_nodes.get_device_ptr_const ()),
    m_bounds (bvh.m_bounds), m_aabb_ids (bvh.m_aabb_ids.get_device_ptr_const ())
  {
  }
};

} // namespace dray
#endif
