// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_COLOR_BAR_ANNOTATOR_HPP
#define DRAY_COLOR_BAR_ANNOTATOR_HPP

#include <vector>

#include <dray/aabb.hpp>
#include <dray/rendering/framebuffer.hpp>

namespace dray
{

class ColorBarAnnotator
{
protected:
  bool m_verticle;
public:
  ColorBarAnnotator();
  void verticle(bool is_verticle);
  void render(Framebuffer &fb,
              Array<Vec<float32, 4>> colors,
              const Vec<float32,2> &pos,
              const Vec<float32,2> &dims);
};

} // namespace dray
#endif
