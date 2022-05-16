// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_CAMERA_HPP
#define DRAY_CAMERA_HPP

#include <dray/aabb.hpp>
#include <dray/matrix.hpp>
#include <dray/ray.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class Camera
{

  protected:
  int32 m_height;
  int32 m_width;
  int32 m_subset_width;
  int32 m_subset_height;
  int32 m_subset_min_x;
  int32 m_subset_min_y;
  float32 m_fov_x;
  float32 m_fov_y;
  float32 m_zoom;

  Vec<float32, 3> m_look;
  Vec<float32, 3> m_up;
  Vec<float32, 3> m_look_at;
  Vec<float32, 3> m_position;

  Array<int32> m_random;
  int32 m_sample;

  void create_rays_imp (Array<Ray> &rays, AABB<> bounds);

  void create_rays_jitter_imp (Array<Ray> &rays, AABB<> bounds);

  Vec<float32,3> m_ray_differential_x;
  Vec<float32,3> m_ray_differential_y;

  public:
  Camera ();

  ~Camera ();

  std::string print () const;

  Vec<float32,3> ray_differential_x() const;
  Vec<float32,3> ray_differential_y() const;

  void reset_to_bounds (const AABB<> bounds,
                        const float64 xpad = 0.,
                        const float64 ypad = 0.,
                        const float64 zpad = 0.);

  void set_height (const int32 &height);

  int32 get_height () const;

  void set_width (const int32 &width);

  int32 get_width () const;

  int32 get_subset_width () const;

  int32 get_subset_height () const;

  void set_fov (const float32 &degrees);

  float32 get_fov () const;

  void set_up (const Vec<float32, 3> &up);

  void set_pos (const Vec<float32, 3> &position);

  Vec<float32, 3> get_pos () const;

  Vec<float32, 3> get_up () const;

  void set_look_at (const Vec<float32, 3> &look_at);

  void set_zoom(const float32 zoom);

  Vec<float32, 3> get_look_at () const;

  void create_rays (Array<Ray> &rays, AABB<> bounds = AABB<> ());

  void create_rays_jitter (Array<Ray> &rays, AABB<> bounds = AABB<> ());

  void trackball_rotate (float32 startX, float32 startY, float32 endX, float32 endY);

  void elevate (const float32 degrees);
  void azimuth (const float32 degrees);

  Matrix<float32, 4, 4> projection_matrix (const float32 near, const float32 far) const;
  Matrix<float32, 4, 4> projection_matrix (const AABB<3> bounds) const;
  Matrix<float32, 4, 4> view_matrix () const;

  void gen_perspective (Array<Ray> &rays);

  int32 subset_size(AABB<3> bounds);

  void gen_perspective_jitter (Array<Ray> &rays);

  Array<float32> gl_depth(const Array<float32> &world_depth, const float32 near, const float32 far);

 // in place transform from gl to world
 void gl_to_world_depth(Array<float32> &gl_depth,
                        const float32 near,
                        const float32 far);

}; // class camera

} // namespace dray
#endif
