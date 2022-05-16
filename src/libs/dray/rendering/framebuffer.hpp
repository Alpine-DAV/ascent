// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FRAMEBUFFER_HPP
#define DRAY_FRAMEBUFFER_HPP

#include <dray/aabb.hpp>
#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{


class Framebuffer
{
  protected:
  Array<Vec<float32, 4>> m_colors;
  Array<float32> m_depths;
  int32 m_width;
  int32 m_height;
  Vec<float32, 4> m_bg_color;
  Vec<float32, 4> m_fg_color;

  public:
  Framebuffer ();
  Framebuffer (const int32 width, const int32 height);

  int32 width () const;
  int32 height () const;

  void clear (); // zero out the color buffer, set depths to inf
  void clear (const Vec<float32, 4> &color);
  void composite_background ();
  void save (const std::string name);
  void save_depth (const std::string name);

  void background_color (const Vec<float32, 4> &color);
  void foreground_color (const Vec<float32, 4> &color);

  Vec<float32, 4> foreground_color () const;
  Vec<float32, 4> background_color () const;

  Array<Vec<float32,4>> colors() const;
  Array<float32> depths() const;


  friend struct DeviceFramebuffer;
};

} // namespace dray
#endif
