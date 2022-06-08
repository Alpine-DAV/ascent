// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SLICE_PLANE_HPP
#define DRAY_SLICE_PLANE_HPP

#include <dray/rendering/traceable.hpp>

namespace dray
{

class SlicePlane : public Traceable
{
  Vec<float32,3> m_point;
  Vec<float32,3> m_normal;
public:
  SlicePlane() = delete;
  SlicePlane(Collection &collection);
  virtual ~SlicePlane();

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) override;
  virtual Array<Fragment> fragments(Array<RayHit> &hits) override;

  void point(const Vec<float32,3> &point);
  void normal(const Vec<float32,3> &normal);
  Vec<float32,3> point() const;
  Vec<float32,3> normal() const;


};

};//namespace dray

#endif//DRAY_VOLUME_INTEGRATOR_HPP
