// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RAY_HIT_HPP
#define DRAY_RAY_HIT_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class RayHit
{
  public:
  int32 m_hit_idx;        /*!< Hit index of primitive hit by ray. -1 means miss */
  Float m_dist;           /*!< Distance to the hit */
  Vec<Float, 3> m_ref_pt; /*!< Refence space coordinates of hit */

  DRAY_EXEC void init ()
  {
    m_hit_idx = -1;
    m_dist = infinity<Float> ();
  }
};

std::ostream &operator<< (std::ostream &out, const RayHit &hit);

} // namespace dray
#endif
