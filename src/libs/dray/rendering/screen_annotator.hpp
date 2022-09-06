// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SS_ANNOTATOR_HPP
#define DRAY_SS_ANNOTATOR_HPP

#include <vector>

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/matrix.hpp>
#include <dray/aabb.hpp>
#include <dray/color_map.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/renderer.hpp>
#include <dray/rendering/rasterbuffer.hpp>

namespace dray
{

class ScreenAnnotator
{
protected:
  std::vector<AABB<2>> m_color_bar_pos;
  int32 m_max_color_bars;
public:
  ScreenAnnotator();

  void max_color_bars(int max_bars);

  void draw_color_bars(Framebuffer &fb,
                       const std::vector<std::string> &field_names,
                       std::vector<ColorMap> &color_maps);

  void draw_triad(Framebuffer &fb,
                  Vec<float32, 2> pos, // screen space coords where we want the triad to be centered (-1 1)
                  float32 distance,
                  Camera &camera);

};

} // namespace dray
#endif
