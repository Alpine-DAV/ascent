// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BILLBOARD_HPP
#define DRAY_BILLBOARD_HPP

#include <vector>

#include <dray/exports.hpp>
#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/camera.hpp>

namespace dray
{

class Billboard
{
protected:
  // note: these are Float to be consistent with the ray class
  Vec<float32,3> m_up;
  Vec<Float,3> m_ray_differential_x;
  Vec<Float,3> m_ray_differential_y;
  Vec<float32,3> m_text_color;
  Array<Vec<float32,3>> m_anchors;
  Array<Vec<float32,2>> m_offsets;
  Array<Vec<float32,2>> m_dims; // width and height of each billboard
  Array<Vec<float32,2>> m_tcoords;
  Array<float32> m_texture;
  int32 m_texture_width;
  int32 m_texture_height;
  BVH m_bvh;
public:
  Billboard() = delete;
  // text: the text of each billboard
  // anchor: world space billboard anchor point
  // offset : the reference space position of the anchor
  //          relative to the billboard center in the [0,1]
  //          range. (0.5, 0.5) will have the billboard pivot
  //          around the center and (0,0) will make it
  //          pivot around the bottom left
  // world_size: how big in world space the billboard is (relative)
  Billboard(const std::vector<std::string> &texts,
            const std::vector<Vec<float32,3>> &anchors,
            const std::vector<Vec<float32,2>> &offsets,
            const std::vector<float32> &world_sizes);

  void camera(const Camera& camera);
  void text_color(const Vec<float32,3> &color);
  Array<RayHit> intersect (const Array<Ray> &rays);
  void shade(const Array<Ray> &rays, const Array<RayHit> &hits, Framebuffer &fb);
  AABB<3> bounds();
  void save_texture(const std::string file_name);

  friend struct DeviceBillboard;
};


} // namespace dray
#endif
