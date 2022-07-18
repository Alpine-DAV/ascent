// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/ray_hit.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const RayHit &hit)
{
  out << "[ hit_idx: " << hit.m_hit_idx << " dist: " << hit.m_dist << " ref "
      << hit.m_ref_pt << " ]";
  return out;
}

} // namespace dray
