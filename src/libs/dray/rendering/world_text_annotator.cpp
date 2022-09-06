// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/world_text_annotator.hpp>
#include <dray/rendering/billboard.hpp>
#include <dray/rendering/colors.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>

#include<cmath>

namespace dray
{

WorldTextAnnotator::WorldTextAnnotator()
  : m_color({0.f,0.f,0.f})
{
}

void
WorldTextAnnotator::clear()
{
  m_texts.clear();
  m_anchors.clear();
  m_offsets.clear();
  m_sizes.clear();
}

void
WorldTextAnnotator::add_text(const std::string text,
                             const Vec<float32,3> &anchor,
                             const Vec<float32,2> &offset,
                             const float32 size)
{
  m_texts.push_back(text);
  m_anchors.push_back(anchor);
  m_offsets.push_back(offset);
  m_sizes.push_back(size);
}



void
WorldTextAnnotator::render(const Camera &camera, Array<Ray> &rays, Framebuffer &fb)
{
  Billboard billboards(m_texts, m_anchors, m_offsets,  m_sizes);
  billboards.camera(camera);
  billboards.text_color(m_color);
  Array<RayHit> hits = billboards.intersect(rays);
  billboards.shade(rays, hits, fb);
}

} // namespace dray
