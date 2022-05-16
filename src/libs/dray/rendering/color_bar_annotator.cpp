// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/color_bar_annotator.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>

#include<cmath>

namespace dray
{


ColorBarAnnotator::ColorBarAnnotator()
  : m_verticle(true)
{
}

void
ColorBarAnnotator::verticle(bool is_verticle)
{
  m_verticle = is_verticle;
}

void
ColorBarAnnotator::render(Framebuffer &fb,
                          Array<Vec<float32, 4>> colors,
                          const Vec<float32,2> &pos,
                          const Vec<float32,2> &dims)
{
  const int32 width = fb.width();
  const int32 height = fb.height();
  const int32 box_width = dims[0];
  const int32 box_height = dims[1];
  const int32 box_size = box_width * box_height;
  const Vec<float32,2> position = pos;
  bool verticle = m_verticle;

  const Vec<float32,4> *color_ptr = colors.get_device_ptr_const();
  const int32 color_size = colors.size();

  DeviceFramebuffer d_framebuffer(fb);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, box_size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 local_x = i % box_width;
    const int32 local_y = i / box_width;
    const int32 x = local_x + int32(position[0]);
    const int32 y = local_y + int32(position[1]);
    const int32 pixel_id = y * width + x;

    if(x >= width || x < 0 || y >= height || y < 0)
    {
      return;
    }

    float32 normalized_height = 0.f;
    if(verticle)
    {
      normalized_height = float32(local_x) / float32(box_width);
    }
    else
    {
      normalized_height = float32(local_y) / float32(box_height);
    }
    int32 lookup_index = 0;
    if(verticle)
    {
     lookup_index = (float32(local_y)*float32(color_size)) / float32(box_height);
    }
    else
    {
      lookup_index = (float32(local_x)*float32(color_size)) / float32(box_width);
    }
    Vec<float32,4> color = color_ptr[lookup_index];

    // draw the transfer function on top of the color map
    // as a grey mask
    if(color[3] < 1.f && normalized_height < color[3])
    {
      color[3] = 1.f;
      Vec<float32,4> mask;
      constexpr float32 intensity = 0.5f;
      mask[0] = intensity;
      mask[1] = intensity;
      mask[2] = intensity;
      mask[3] = intensity;
      blend(mask, color);
      color = mask;
    }
    color[3] = 1.f;
    d_framebuffer.m_colors[pixel_id] = color;

  });
  DRAY_ERROR_CHECK();

}

} // namespace dray
