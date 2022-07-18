// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/rendering/slice_plane.hpp>
#include <dray/error_check.hpp>
#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/data_model/device_field.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{

Array<RayHit>
get_hits(const Array<Ray> &rays,
         const Array<Location> &locations,
         const Array<Vec<Float,3>> &points)
{
  Array<RayHit> hits;
  hits.resize(rays.size());

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const Location *loc_ptr = locations.get_device_ptr_const();
  const Vec<Float,3> *points_ptr = points.get_device_ptr_const();
  RayHit *hit_ptr = hits.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, rays.size()), [=] DRAY_LAMBDA (int32 i)
  {
    RayHit hit;
    const Location loc = loc_ptr[i];
    const Ray ray = ray_ptr[i];
    const Vec<Float,3> point = points_ptr[i];
    hit.m_hit_idx = loc.m_cell_id;
    hit.m_ref_pt  = loc.m_ref_pt;

    if(hit.m_hit_idx > -1)
    {
      hit.m_dist = (point - ray.m_orig).magnitude();
    }

    hit_ptr[i] = hit;

  });
  DRAY_ERROR_CHECK();
  return hits;
}

template<typename ElementType>
Array<Fragment>
get_fragments(UnstructuredField<ElementType> &field,
              Array<RayHit> &hits,
              Vec<float32,3> normal)
{
  const int32 size = hits.size();

  Array<Fragment> fragments;
  fragments.resize(size);
  Fragment *fragment_ptr = fragments.get_device_ptr();

  const RayHit *hit_ptr = hits.get_device_ptr_const();

  DeviceField<ElementType> device_field(field);
//  #warning "unify fragment and ray hit initialization"
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    Fragment frag;
    // TODO: create struct initializers
    frag.m_normal = normal;
    frag.m_scalar= 3.14f;

    const RayHit &hit = hit_ptr[i];

    if (hit.m_hit_idx > -1)
    {
      // Evaluate element transformation to get scalar field value and gradient.

      const int32 el_id = hit.m_hit_idx;

      Vec<Vec<Float,1>,3> field_deriv;
      Vec<Float,1> scalar;
      scalar = device_field.get_elem(el_id).eval_d(hit.m_ref_pt, field_deriv);
      frag.m_scalar = scalar[0];
    }

    fragment_ptr[i] = frag;

  });
  DRAY_ERROR_CHECK();

  return fragments;
}

Array<Vec<Float,3>>
calc_sample_points(Array<Ray> &rays,
                   const Vec<float32,3> &point,
                   const Vec<float32,3> &normal)
{
  const int32 size = rays.size();

  Array<Vec<Float,3>> points;
  points.resize(size);

  Vec<Float,3> t_normal;
  t_normal[0] = normal[0];
  t_normal[1] = normal[1];
  t_normal[2] = normal[2];

  Vec<Float,3> t_point;
  t_point[0] = point[0];
  t_point[1] = point[1];
  t_point[2] = point[2];

  Vec<Float,3> *points_ptr = points.get_device_ptr();

  const Ray *ray_ptr = rays.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray &ray = ray_ptr[i];
    const Float denom = dot(ray.m_dir, t_normal);
    Float dist = infinity<Float>();
    if(abs(denom) > 1e-6)
    {
      Vec<Float,3> p = t_point - ray.m_orig;
      const Float t = dot(p, t_normal) / denom;
      if(t > 0 && t < ray.m_far && t > ray.m_near)
      {
        dist = t;
      }
    }

    Vec<Float,3> sample = ray.m_dir * dist + ray.m_orig;

    points_ptr[i] = sample;

  });
  DRAY_ERROR_CHECK();

  return points;
}

template<class Element>
Array<RayHit>
slice_execute(UnstructuredMesh<Element> &mesh,
              Array<Ray> &rays,
              const Vec<float32,3> point,
              const Vec<float32,3> normal)
{
  DRAY_LOG_OPEN("slice_plane");

  Array<Vec<Float,3>> samples = detail::calc_sample_points(rays, point, normal);

  // Find elements and reference coordinates for the points.
  Array<Location> locations = mesh.locate(samples);

  Array<RayHit> hits = detail::get_hits(rays, locations, samples);

  DRAY_LOG_CLOSE();
  return hits;
}


struct SliceFunctor
{
  Array<Ray> *m_rays;
  Array<RayHit> m_hits;
  Vec<float32,3> m_point;
  Vec<float32,3> m_normal;
  SliceFunctor(Array<Ray> *rays,
               const Vec<float32,3> point,
               const Vec<float32,3> normal )
    : m_rays(rays),
      m_point(point),
      m_normal(normal)
  {
  }

  template<typename MeshType>
  void operator()(MeshType &mesh)
  {
    m_hits = slice_execute(mesh, *m_rays, m_point, m_normal);
  }
};

struct SliceFragmentFunctor
{
  SlicePlane *m_slicer;
  Array<RayHit> *m_hits;
  Array<Fragment> m_fragments;
  SliceFragmentFunctor(SlicePlane  *slicer,
                  Array<RayHit> *hits)
    : m_slicer(slicer),
      m_hits(hits)
  {
  }

  template<typename FieldType>
  void operator()(FieldType &field)
  {
    m_fragments = detail::get_fragments(field, *m_hits, m_slicer->normal());
  }
};

} // namespace detail

SlicePlane::SlicePlane(Collection &collection)
  : Traceable(collection)
{
  m_point[0] = 0.f;
  m_point[1] = 0.f;
  m_point[2] = 0.f;

  m_normal[0] = 0.f;
  m_normal[1] = 1.f;
  m_normal[2] = 0.f;
}

SlicePlane::~SlicePlane()
{
}


Array<RayHit>
SlicePlane::nearest_hit(Array<Ray> &rays)
{
  DataSet data_set = m_collection.domain(m_active_domain);
  Mesh *mesh = data_set.mesh();

  detail::SliceFunctor func(&rays, m_point, m_normal);
  dispatch_3d(mesh, func);
  return func.m_hits;
}

Array<Fragment>
SlicePlane::fragments(Array<RayHit> &hits)
{
  DRAY_LOG_OPEN("fragments");
  assert(m_field_name != "");

  DataSet data_set = m_collection.domain(m_active_domain);
  Field *field = data_set.field(m_field_name);

  detail::SliceFragmentFunctor func(this,&hits);
  dispatch_3d_scalar(field, func);
  DRAY_LOG_CLOSE();
  return func.m_fragments;
}

void
SlicePlane::point(const Vec<float32,3> &point)
{
  m_point = point;
}

Vec<float32,3>
SlicePlane::point() const
{
  return m_point;
}

void
SlicePlane::normal(const Vec<float32,3> &normal)
{
  m_normal = normal;
  m_normal.normalize();
}

Vec<float32,3>
SlicePlane::normal() const
{
  return m_normal;
}

}//namespace dray

