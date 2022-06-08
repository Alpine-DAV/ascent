// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRACEABLE_HPP
#define DRAY_TRACEABLE_HPP

#include <dray/array.hpp>
#include <dray/color_map.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/fragment.hpp>
#include <dray/rendering/point_light.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/data_model/collection.hpp>

namespace dray
{
/**
 * \class Traceable
 * \brief Encapsulates a traceable object
 *
 * Defines the interface for traceable objects
 */

class Traceable
{
protected:
  Collection m_collection;
  std::string m_field_name;
  ColorMap m_color_map;
  int32 m_active_domain;
  Range m_field_range;
public:
  Traceable() = delete;
  Traceable(Collection &collection);
  virtual ~Traceable();
  /// returns the nearests hit along a batch of rays
  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) = 0;
  /// returns the fragments for a batch of hits
  virtual Array<Fragment> fragments(Array<RayHit> &hits);

  // shading with lighting
  virtual void shade(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     const Array<PointLight> &lights,
                     Framebuffer &framebuffer);

  // shade without lighting
  virtual void shade(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     Framebuffer &framebuffer);

  // get the raw colors
  virtual void colors(const Array<Ray> &rays,
                      const Array<RayHit> &hits,
                      const Array<Fragment> &fragments,
                      Array<Vec<float32,4>> &colors);

  void active_domain(int32 domain_index);
  int32 active_domain();
  int32 num_domains();

  /// set the input collection
  void input(Collection &collection);
  /// sets the field for that generates fragments for shading
  void field(const std::string &field_name);
  std::string field() const;
  void color_map(ColorMap &color_map);

  ColorMap& color_map();
  Collection& collection();
};


} // namespace dray
#endif
