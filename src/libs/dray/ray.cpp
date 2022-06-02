// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/array_utils.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <dray/ray.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const Ray &r)
{
  out << r.m_pixel_id;
  return out;
}

Array<Vec<Float, 3>> calc_tips (const Array<Ray> &rays, const Array<RayHit> &hits)
{
  const int32 ray_size = rays.size ();
  const int32 hit_size = hits.size ();
  (void) hit_size;
  assert (ray_size == hit_size);

  Array<Vec<Float, 3>> tips;
  tips.resize (ray_size);

  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const RayHit *hit_ptr = hits.get_device_ptr_const ();

  Vec<Float, 3> *tips_ptr = tips.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, ray_size), [=] DRAY_LAMBDA (int32 ii) {
    Ray ray = ray_ptr[ii];
    RayHit hit = hit_ptr[ii];
    Vec<Float, 3> point = { infinity<Float> (), infinity<Float> (), infinity<Float> () };
    if (hit.m_hit_idx != -1)
    {
      point = ray.m_orig + ray.m_dir * hit.m_dist;
    }
    tips_ptr[ii] = point;
  });
  DRAY_ERROR_CHECK();

  return tips;
}

Array<int32> active_indices (const Array<Ray> &rays, const Array<RayHit> &hits)
{
  const int32 ray_size = rays.size ();
  const int32 hit_size = hits.size ();
  (void) hit_size;
  assert (hit_size == ray_size);

  Array<int32> active_flags;
  active_flags.resize (ray_size);

  const Ray *ray_ptr = rays.get_device_ptr_const ();
  const RayHit *hit_ptr = hits.get_device_ptr_const ();

  int32 *flags_ptr = active_flags.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, ray_size), [=] DRAY_LAMBDA (int32 ii) {
    uint8 flag =
    ((hit_ptr[ii].m_hit_idx > -1) && (ray_ptr[ii].m_near < ray_ptr[ii].m_far) &&
     (hit_ptr[ii].m_dist < ray_ptr[ii].m_far)) ?
    1 :
    0;
    flags_ptr[ii] = flag;
  });
  DRAY_ERROR_CHECK();

  return index_flags (active_flags);
}

void advance_ray (Array<Ray> &rays, float32 distance)
{
  // avoid lambda capture issues
  Float dist = distance;

  Ray *ray_ptr = rays.get_device_ptr ();

  const int32 size = rays.size ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    Ray &ray = ray_ptr[i];
    // advance ray
    ray.m_near += dist;
  });
}

void cull_missed_rays (Array<Ray> &rays, AABB<> bounds)
{
  Array<RayHit> hits;
  calc_ray_start (rays, hits, bounds);
  Array<int32> active_rays = active_indices (rays, hits);
  rays = gather (rays, active_rays);
}

Array<Ray> remove_missed_rays (Array<Ray> &rays, AABB<> bounds)
{
  Array<Ray> res;
  if(!bounds.is_empty())
  {
    Array<int32> active_rays = mark_active(rays, bounds);
    Array<int32> idxs = index_flags (active_rays);
    res = gather (rays, idxs);
    // TODO: using hits is kinda wastefull
    Array<RayHit> hits;
    calc_ray_start (res, hits, bounds);
  }
  return res;
}

Array<int32> mark_active(Array<Ray> &rays, AABB<> bounds)
{
  // avoid lambda capture issues
  AABB<> mesh_bounds = bounds;
  // be conservative
  mesh_bounds.scale (1.001f);

  const Ray *ray_ptr = rays.get_device_ptr_const();

  const int32 size = rays.size ();

  Array<int32> active;
  active.resize(size);
  int32 * active_ptr = active.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {

    Ray ray = ray_ptr[i];
    int32 active = 0;

    const Vec<Float, 3> ray_dir = ray.m_dir;
    const Vec<Float, 3> ray_orig = ray.m_orig;

    float32 dirx = static_cast<float32> (ray_dir[0]);
    float32 diry = static_cast<float32> (ray_dir[1]);
    float32 dirz = static_cast<float32> (ray_dir[2]);
    float32 origx = static_cast<float32> (ray_orig[0]);
    float32 origy = static_cast<float32> (ray_orig[1]);
    float32 origz = static_cast<float32> (ray_orig[2]);

    const float32 inv_dirx = rcp_safe (dirx);
    const float32 inv_diry = rcp_safe (diry);
    const float32 inv_dirz = rcp_safe (dirz);

    const float32 odirx = origx * inv_dirx;
    const float32 odiry = origy * inv_diry;
    const float32 odirz = origz * inv_dirz;

    const float32 xmin = mesh_bounds.m_ranges[0].min () * inv_dirx - odirx;
    const float32 ymin = mesh_bounds.m_ranges[1].min () * inv_diry - odiry;
    const float32 zmin = mesh_bounds.m_ranges[2].min () * inv_dirz - odirz;
    const float32 xmax = mesh_bounds.m_ranges[0].max () * inv_dirx - odirx;
    const float32 ymax = mesh_bounds.m_ranges[1].max () * inv_diry - odiry;
    const float32 zmax = mesh_bounds.m_ranges[2].max () * inv_dirz - odirz;

    const float32 min_int = ray.m_near;
    float32 min_dist =
    max (max (max (min (ymin, ymax), min (xmin, xmax)), min (zmin, zmax)), min_int);
    float32 max_dist = min (min (max (ymin, ymax), max (xmin, xmax)), max (zmin, zmax));
    max_dist = min(max_dist, float32(ray.m_far));

    if (max_dist > min_dist)
    {
      active = 1;
    }
    active_ptr[i] = active;

  });
  DRAY_ERROR_CHECK();
  return active;
}

void calc_ray_start (Array<Ray> &rays, Array<RayHit> &hits, AABB<> bounds)
{
  // avoid lambda capture issues
  AABB<> mesh_bounds = bounds;
  // be conservative
  mesh_bounds.scale (1.001f);

  Ray *ray_ptr = rays.get_device_ptr ();

  const int32 size = rays.size ();

  hits.resize (size);
  RayHit *hit_ptr = hits.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    Ray ray = ray_ptr[i];
    RayHit hit = hit_ptr[i];
    const Vec<Float, 3> ray_dir = ray.m_dir;
    const Vec<Float, 3> ray_orig = ray.m_orig;

    float32 dirx = static_cast<float32> (ray_dir[0]);
    float32 diry = static_cast<float32> (ray_dir[1]);
    float32 dirz = static_cast<float32> (ray_dir[2]);
    float32 origx = static_cast<float32> (ray_orig[0]);
    float32 origy = static_cast<float32> (ray_orig[1]);
    float32 origz = static_cast<float32> (ray_orig[2]);

    const float32 inv_dirx = rcp_safe (dirx);
    const float32 inv_diry = rcp_safe (diry);
    const float32 inv_dirz = rcp_safe (dirz);

    const float32 odirx = origx * inv_dirx;
    const float32 odiry = origy * inv_diry;
    const float32 odirz = origz * inv_dirz;

    const float32 xmin = mesh_bounds.m_ranges[0].min () * inv_dirx - odirx;
    const float32 ymin = mesh_bounds.m_ranges[1].min () * inv_diry - odiry;
    const float32 zmin = mesh_bounds.m_ranges[2].min () * inv_dirz - odirz;
    const float32 xmax = mesh_bounds.m_ranges[0].max () * inv_dirx - odirx;
    const float32 ymax = mesh_bounds.m_ranges[1].max () * inv_diry - odiry;
    const float32 zmax = mesh_bounds.m_ranges[2].max () * inv_dirz - odirz;

    const float32 min_int = ray.m_near;
    float32 min_dist =
    max (max (max (min (ymin, ymax), min (xmin, xmax)), min (zmin, zmax)), min_int);
    float32 max_dist = min (min (max (ymin, ymax), max (xmin, xmax)), max (zmin, zmax));
    max_dist = min(max_dist, float32(ray.m_far));

    hit.m_hit_idx = -1;
    if (max_dist > min_dist)
    {
      hit.m_hit_idx = 0; // just give this a dummy value that is a valid hit
    }

    ray.m_near = min_dist;
    hit.m_dist = min_dist;
    ray.m_far = max_dist;

    ray_ptr[i] = ray;
    hit_ptr[i] = hit;
  });
  DRAY_ERROR_CHECK();
}

void ray_max(Array<Ray> &rays, const Array<RayHit> &hits)
{
  const int32 size = rays.size();
  Ray *ray_ptr = rays.get_device_ptr();
  const RayHit *hit_ptr = hits.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const RayHit hit = hit_ptr[i];
    if(hit.m_hit_idx != -1)
    {
      ray_ptr[i].m_far = hit.m_dist;
    }

  });
  DRAY_ERROR_CHECK();
}

} // namespace dray
