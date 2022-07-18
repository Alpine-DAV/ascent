// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SURFACE_HPP
#define DRAY_SURFACE_HPP

#include<dray/rendering/traceable.hpp>

namespace dray
{

class Surface : public Traceable
{
protected:
  bool m_draw_mesh;
  float32 m_line_thickness;
  float32 m_sub_res; // sub resolution of grid lines
  Vec<float32,4> m_line_color;

public:
  Surface() = delete;
  Surface(Collection &collection);
  virtual ~Surface();

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) override;

  virtual void shade(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     const Array<PointLight> &lights,
                     Framebuffer &framebuffer) override;

  virtual void shade(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     Framebuffer &framebuffer) override;

  virtual void colors(const Array<Ray> &rays,
                     const Array<RayHit> &hits,
                     const Array<Fragment> &fragments,
                     Array<Vec<float32,4>> &colors) override;

  void draw_mesh(const Array<Ray> &rays,
                 const Array<RayHit> &hits,
                 const Array<Fragment> &fragments,
                 Framebuffer &framebuffer);

  void draw_mesh(const Array<Ray> &rays,
                 const Array<RayHit> &hits,
                 const Array<Fragment> &fragments,
                 Array<Vec<float32,4>> &colors);

  void draw_mesh(bool on);
  void mesh_sub_res(float32 sub_res);
  void line_thickness(const float32 thickness);
  void line_color(const Vec<float32,4> &color);
};

};//namespace dray

#endif //DRAY_SURFACE_HPP
