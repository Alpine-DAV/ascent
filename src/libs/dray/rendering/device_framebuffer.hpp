// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_FRAMEBUFFER_HPP
#define DRAY_DEVICE_FRAMEBUFFER_HPP

#include <dray/rendering/framebuffer.hpp>

namespace dray
{

struct DeviceFramebuffer
{
  Vec<float32, 4> *m_colors;
  float32 *m_depths;

  DeviceFramebuffer () = delete;

  DeviceFramebuffer (Framebuffer &framebuffer)
  {
    m_colors = framebuffer.m_colors.get_device_ptr ();
    m_depths = framebuffer.m_depths.get_device_ptr ();
  }

  void DRAY_EXEC set_color (const int32 &index, const Vec<float32, 4> &color)
  {
    m_colors[index] = color;
  }

  void DRAY_EXEC set_depth (const int32 &index, const float32 &depth)
  {
    m_depths[index] = depth;
  }
};

} // namespace dray
#endif
