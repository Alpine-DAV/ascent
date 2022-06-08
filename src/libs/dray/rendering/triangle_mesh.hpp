// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TRIANGLE_MESH_HPP
#define DRAY_TRIANGLE_MESH_HPP

#include <dray/aabb.hpp>
#include <dray/array.hpp>
#include <dray/rendering/fragment.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/texture2d.hpp>
#include <dray/rendering/material.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/ray.hpp>
#include <vector>

namespace dray
{

class TriangleMesh
{
  protected:
  Array<Vec<float32,3>> m_coords;
  Array<Vec<int32,3>> m_indices;

  // TODO: should we subclass this?
  // in the subclass world the obj reader would be a
  // mesh factory and spit out a non-textured mesh or a
  // textured one
  Array<Vec<float32,2>> m_tcoords;
  Array<Vec<int32,3>> m_tindices;
  std::vector<Texture2d> m_textures;

  Array<Material> m_materials;
  Array<int32> m_mat_ids;

  BVH m_bvh;

  TriangleMesh ();

  public:
  TriangleMesh (const std::string obj_file);
  TriangleMesh (Array<Vec<float32,3>> &coords, Array<Vec<int32,3>> &indices);
  ~TriangleMesh ();

  Array<RayHit> intersect (const Array<Ray> &rays);

  void shade(const Array<Ray> &rays, const Array<RayHit> &hits, Framebuffer &fb);
  //Array<Fragment>
  //fragments(const Array<Ray> &rays, const Array<RayHit> &hits);
  void write(const Array<Ray> &rays, const Array<RayHit> &hits, Framebuffer &fb);

  Array<Vec<float32,3>> &coords ();
  Array<Vec<int32,3>> &indices ();
  AABB<> get_bounds ();
  bool is_textured();
};

} // namespace dray

#endif
