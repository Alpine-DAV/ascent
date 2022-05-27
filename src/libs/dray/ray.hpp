// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RAY_HPP
#define DRAY_RAY_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array.hpp>
#include <dray/ray_hit.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
// TODO: includes that need to be moved to ray ops
#include <dray/aabb.hpp>

namespace dray
{

class Ray
{
  public:
  Vec<Float, 3> m_dir;
  Float m_near;
  Vec<Float, 3> m_orig;
  Float m_far;
  int32 m_pixel_id;
};

std::ostream &operator<< (std::ostream &out, const Ray &r);

/*! \brief Calculate the point at the current distance along the ray
 *         If the ray missed, this point is set to inf
 */
Array<Vec<Float, 3>> calc_tips (const Array<Ray> &rays, const Array<RayHit> &hits);

Array<int32> active_indices (const Array<Ray> &rays, const Array<RayHit> &hits);

void advance_ray (Array<Ray> &rays, float32 distance);

//
// utility function to find estimated intersection with the bounding
// box of the mesh
//
// After calling:
//   rays m_near   : set to estimated mesh entry
//   rays m_far    : set to estimated mesh exit
//
//   if ray missed then m_far <= m_near, if ray hit then m_far > m_near.
//
void calc_ray_start (Array<Ray> &rays, Array<RayHit> &hits, AABB<> bounds);

// set the hit index of rays that hit the bounding box
Array<int32> mark_active(Array<Ray> &rays, AABB<> bounds);

// TODO: this should ultimately return a subset of rays and
// leave the input as is in order to support a broader set of
// compositions. Fun
// remove any rays that miss the bounds
// After calling:
//   only the rays that hit the bounds will remain
//   rays m_near   : set to estimated mesh entry
//   rays m_far    : set to estimated mesh exit
void cull_missed_rays (Array<Ray> &rays, AABB<> bounds);
Array<Ray> remove_missed_rays (Array<Ray> &rays, AABB<> bounds);

// set ray max distance to hit distances
void ray_max(Array<Ray> &rays, const Array<RayHit> &hits);

} // namespace dray
#endif
