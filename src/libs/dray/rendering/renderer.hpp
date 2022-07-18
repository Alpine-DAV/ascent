// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RENDERER_HPP
#define DRAY_RENDERER_HPP

#include <dray/rendering/camera.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/point_light.hpp>
#include <dray/rendering/traceable.hpp>
#include <dray/rendering/volume.hpp>

#include <memory>
#include <vector>

namespace dray
{

class Renderer
{
protected:
  std::vector<std::shared_ptr<Traceable>> m_traceables;
  std::shared_ptr<Volume> m_volume;
  std::vector<PointLight> m_lights;
  bool m_use_lighting;
  bool m_world_annotations;
  bool m_color_bar;
  bool m_triad;
  int32 m_max_color_bars;

public:
  Renderer();
  void clear();
  void clear_lights();
  AABB<3> bounds();
  void add(std::shared_ptr<Traceable> traceable);
  void volume(std::shared_ptr<Volume> volume);
  void add_light(const PointLight &light);
  void use_lighting(bool use_it);
  Framebuffer render(Camera &camera);
  void composite(Array<Ray> &rays,
                 Camera &camera,
                 Framebuffer &framebuffer,
                 bool synch_deptsh) const;

  void color_bar(bool on);
  void triad(bool on);
  void world_annotations(bool on);
  void max_color_bars(const int32 max_bars);

};


} // namespace dray
#endif
