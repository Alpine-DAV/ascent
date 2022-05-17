// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRIANGLE_INTERSECTION_HPP
#define DRAY_TRIANGLE_INTERSECTION_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/ray_hit.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <typename IntersectorType> class TriLeafIntersector
{
  public:
  template <typename T>
  DRAY_EXEC void intersect_leaf (const int32 &leaf_index,
                                 const Vec<T, 3> &orig,
                                 const Vec<T, 3> &dir,
                                 RayHit &hit,
                                 T &closest_dist,
                                 const T &min_dist,
                                 const Vec<int32,3> *indices,
                                 const Vec<float32,3> *points,
                                 const int32 *leafs) const
  {
    const int32 tri = leafs[leaf_index];
    Vec3i vids = indices[tri];

    Vec<T, 3> vertices[3];
    for (int32 i = 0; i < 3; ++i)
    {
      vertices[i] = points[vids[i]];
    }


    IntersectorType intersector;
    T distance = -1.;
    T u, v;

    intersector.intersect (vertices[0],
                           vertices[1],
                           vertices[2],
                           dir,
                           orig,
                           distance,
                           u,
                           v);

    if (distance != -1. && distance < closest_dist && distance > min_dist)
    {
      closest_dist = distance;
      hit.m_dist = distance;
      hit.m_ref_pt[0] = u;
      hit.m_ref_pt[1] = v;
      hit.m_ref_pt[2] = 1.f - u - v;
      hit.m_hit_idx = leafs[leaf_index];
    }
  }
};

class Moller
{
  public:
  template <typename T>
  DRAY_EXEC void intersect (const Vec<T, 3> &a,
                            const Vec<T, 3> &b,
                            const Vec<T, 3> &c,
                            const Vec<T, 3> &dir,
                            const Vec<T, 3> &orig,
                            T &distance,
                            T &u,
                            T &v) const
  {
    constexpr float32 epsilon = 0.000001f;

    Vec<T, 3> e1 = b - a;
    Vec<T, 3> e2 = c - a;

    Vec<T, 3> p = cross (dir, e2);

    T pdot = dot (e1, p);
    if (pdot != 0.f)
    {
      pdot = 1.f / pdot;
      Vec<T, 3> t = orig - a;

      u = dot (t, p) * pdot;
      if (u >= (0.f - epsilon) && u <= (1.f + epsilon))
      {

        Vec<T, 3> q = cross (t, e1); // = t % e1;
        v = dot (dir, q) * pdot;
        if (v >= (0.f - epsilon) && v <= (1.f + epsilon) && !(u + v > 1.f + 2 * epsilon))
        {
          distance = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * pdot;
          distance = dot (e2, q) * pdot;
        }
      }
    }
  }

}; // Moller

} // namespace dray
#endif
