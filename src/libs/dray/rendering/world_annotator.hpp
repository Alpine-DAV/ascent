// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_WORLD_ANNOTATOR_HPP
#define DRAY_WORLD_ANNOTATOR_HPP

#include <vector>

#include <dray/aabb.hpp>
#include <dray/color_map.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/camera.hpp>

namespace dray
{

class WorldAnnotator
{
protected:
  AABB<3> m_bounds;
  std::vector<Vec<float32,3>> m_starts;
  std::vector<Vec<float32,3>> m_ends;
  std::vector<std::string> m_annotations;
  std::vector<Vec<float32,3>> m_annot_positions;
  std::vector<float32> m_annot_sizes;

public:
  WorldAnnotator(AABB<3> bounds);
  WorldAnnotator() = delete;

  void add_axes(const Camera &camera);
  void add_bounding_box();

  void render(Framebuffer &fb, Array<Ray> &rays, const Camera &camera);

//  void screen_annotations(Framebuffer &fb,
//                          const std::vector<std::string> &field_names,
//                          std::vector<ColorMap> &color_maps);
//
};

} // namespace dray
#endif
