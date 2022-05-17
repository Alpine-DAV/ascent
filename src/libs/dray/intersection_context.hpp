// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_INTERSECTION_CONTEXT_HPP
#define DRAY_INTERSECTION_CONTEXT_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class IntersectionContext
{
  public:
  int32 m_is_valid;
  Vec<Float, 3> m_hit_pt;
  Vec<Float, 3> m_normal;
  Vec<Float, 3> m_ray_dir;
  int32 m_pixel_id;
};

std::ostream &operator<< (std::ostream &out, const IntersectionContext &r);

} // namespace dray
#endif
