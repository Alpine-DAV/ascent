// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_WORLD_TEXT_ANNOTATOR_HPP
#define DRAY_WORLD_TEXT_ANNOTATOR_HPP

#include <vector>

#include <dray/aabb.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/camera.hpp>

namespace dray
{

class WorldTextAnnotator
{
protected:
  Vec<float32,3> m_color;
  std::vector<std::string> m_texts;
  std::vector<Vec<float32,3>> m_anchors;
  std::vector<Vec<float32,2>> m_offsets;
  std::vector<float32> m_sizes;
public:
  WorldTextAnnotator();
  void clear();
  void add_text(const std::string text,
                const Vec<float32,3> &anchor,
                const Vec<float32,2> &offset,
                const float32 size);

  void render(const Camera &camera, Array<Ray> &rays, Framebuffer &fb);
};

} // namespace dray
#endif
