// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_VOLUME_RENDERER_HPP
#define DRAY_VOLUME_RENDERER_HPP

#include <dray/data_model/collection.hpp>
#include <dray/color_map.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/rendering/volume_partial.hpp>
#include <dray/rendering/point_light.hpp>

namespace dray
{

class Volume
{
protected:
  int32 m_samples;
  ColorMap m_color_map;
  Collection m_collection;
  std::string m_field;
  AABB<3> m_bounds;
  bool m_use_lighting;
  int32 m_active_domain;
  Range m_field_range;

public:
  Volume() = delete;
  Volume(Collection &collection);
  ~Volume();

  void active_domain(int32 domain_index);
  int32 active_domain();
  int32 num_domains();

  Array<VolumePartial> integrate(Array<Ray> &rays, Array<PointLight> &lights);

  void save(const std::string name,
            Array<VolumePartial> partials,
            const int32 width,
            const int32 height);

  /// set the input data set
  void input(Collection &collection);

  /// set the number of samples based on the bounds.
  void samples(int32 num_samples);

  void field(const std::string field);
  std::string field() const;

  void use_lighting(bool do_it);

  ColorMap& color_map();
};


} // namespace dray
#endif
