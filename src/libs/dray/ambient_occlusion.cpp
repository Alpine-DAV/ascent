// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/ambient_occlusion.hpp>
#include <dray/error_check.hpp>
#include <dray/array_utils.hpp>
#include <dray/halton.hpp>
#include <dray/intersection_context.hpp>
#include <dray/math.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>

#include <stdio.h> /* NULL */
#include <time.h> /* time */

//#include <sstream>
namespace dray
{

const Float AmbientOcclusion::nudge_dist = 0.00005f;


Array<Ray> AmbientOcclusion::gen_occlusion (const Array<IntersectionContext> intersection_ctx,
                                            const int32 occ_samples,
                                            const Float occ_near,
                                            const Float occ_far)
{
  Array<int32> unused_array;
  return AmbientOcclusion::gen_occlusion (intersection_ctx, occ_samples,
                                          occ_near, occ_far, unused_array);
}

Array<Ray> AmbientOcclusion::gen_occlusion (const Array<IntersectionContext> intersection_ctx,
                                            const int32 occ_samples,
                                            const Float occ_near,
                                            const Float occ_far,
                                            Array<int32> &compact_indexing)
{
  // Some intersection contexts may represent non-intersections.
  // We only produce occlusion rays for valid intersections.
  // Therefore we re-index the set of rays which actually hit something.
  //   0 .. ray_idx .. (num_prim_rays-1)
  //   0 .. hit_idx .. (num_prim_hits-1)
  const int32 num_prim_rays = intersection_ctx.size ();

  const IntersectionContext *ctx_ptr = intersection_ctx.get_device_ptr_const ();
  Array<int32> flags;
  flags.resize (intersection_ctx.size ());
  int32 *flags_ptr = flags.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, intersection_ctx.size ()),
                            [=] DRAY_LAMBDA (int32 ii) {
                              flags_ptr[ii] = ctx_ptr[ii].m_is_valid;
                            });
  DRAY_ERROR_CHECK();

  int32 num_prim_hits;
  compact_indexing = array_compact_indices (flags, num_prim_hits);

  // Initialize entropy array, needed before sampling Halton hemisphere.
  Array<int32> entropy = array_random (num_prim_hits, time (NULL), num_prim_hits); // TODO choose right upper bound

  // Allocate new occlusion rays.
  Array<Ray> occ_rays;
  occ_rays.resize (num_prim_hits * occ_samples);
  Ray *occ_ray_ptr = occ_rays.get_device_ptr ();

  // "l" == "local": Capture parameters to local variables, for loop kernel.
  const Float l_nudge_dist = AmbientOcclusion::nudge_dist;
  const int32 l_occ_samples = occ_samples;

  // Input pointers.
  const int32 *entropy_ptr = entropy.get_device_ptr_const ();
  const int32 *compact_indexing_ptr = compact_indexing.get_device_ptr_const ();

  // For each incoming hit, generate (occ_samples) new occlusion rays.
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_prim_rays * occ_samples), [=] DRAY_LAMBDA (int32 ii) {
    // We launch (occ_samples) instances for each incoming ray.
    // This thread is identified by two indices:
    //  0 <= prim_ray_idx   < num_prim_rays
    //  0 <= occ_sample_idx < occ_samples
    const int32 prim_ray_idx = ii / l_occ_samples;
    const int32 sample = ii % l_occ_samples;
    const IntersectionContext ctx = ctx_ptr[ii];
    // First test whether the intersection is valid; only proceed if it is.
    if (ctx.m_is_valid)
    {
      // Get normal and construct basis for tangent space.
      // Note: We need to do this for each hit, not just for each intersected element.
      //   Unless the elements are flat (triangles), the surface normal can vary
      //   within a single element, depending on the location of the hit point.
      Vec<Float, 3> tangent_x, tangent_y;
      ConstructTangentBasis (ctx.m_normal, tangent_x, tangent_y);

      // Make a 'nudge vector' to displace occlusion rays, avoid self-intersection.
      /// Vec<T,3> nudge = normal * l_nudge_dist;
      Vec<Float, 3> nudge = ctx.m_ray_dir * (-l_nudge_dist);

      // Find output indices for this sample.
      const int32 prim_hit_idx = compact_indexing_ptr[prim_ray_idx];
      const int32 occ_offset = prim_hit_idx * l_occ_samples;

      // Get Halton hemisphere sample in local coordinates.
      Vec<Float, 3> occ_local_direction =
      CosineWeightedHemisphere (entropy_ptr[prim_hit_idx] + sample);

      // Map these coordinates onto the local frame, get world coordinates.
      Vec<Float, 3> occ_direction = tangent_x * occ_local_direction[0] +
                                    tangent_y * occ_local_direction[1] +
                                    ctx.m_normal * occ_local_direction[2];

      occ_direction.normalize ();

      Ray occ_ray;
      occ_ray.m_near = occ_near;
      occ_ray.m_far = occ_far;
      occ_ray.m_dir = occ_direction;
      occ_ray.m_orig = ctx.m_hit_pt + nudge;
      occ_ray.m_pixel_id = ctx.m_pixel_id;

      occ_ray_ptr[occ_offset + sample] = occ_ray;
    }
  });
  DRAY_ERROR_CHECK();

  return occ_rays;
}

// ----------------------------------------------

// These sampling methods were adapted from https://gitlab.kitware.com/mclarsen/vtk-m/blob/pathtracer/vtkm/rendering/raytracing/Sampler.h
// - CosineWeightedHemisphere
// - ConstructTangentBasis (factored from CosineWeightedHemisphere).
// TODO Convert camelCase (vtk-m) to lower_case (dray) ?

DRAY_EXEC Vec<Float, 3> AmbientOcclusion::CosineWeightedHemisphere (const int32 &sampleNum)
{
  Vec<Float, 2> xy;
  Halton2D<Float, 3> (sampleNum, xy);
  const Float r = sqrt (xy[0]);
  const Float theta = 2 * pi () * xy[1];

  Vec<Float, 3> direction;
  direction[0] = r * cos (theta);
  direction[1] = r * sin (theta);
  direction[2] = sqrt (max (Float(0.0f), Float(1.f) - xy[0]));
  return direction;

  // Vec<T,3> sampleDir;
  // sampleDir[0] = dot(direction, xAxis);
  // sampleDir[1] = dot(direction, yAxis);
  // sampleDir[2] = dot(direction, normal);
  // return sampleDir;
}

DRAY_EXEC void AmbientOcclusion::ConstructTangentBasis (const Vec<Float, 3> &normal,
                                              Vec<Float, 3> &xAxis,
                                              Vec<Float, 3> &yAxis)
{
  // generate orthoganal basis about normal (i.e. basis for tangent space).
  // kz will be the axis idx (0,1,2) most aligned with normal.
  // TODO MAI [2018-05-30] I propose we instead choose the axis LEAST aligned with normal;
  // this amounts to flipping all the > to instead be <.
  int32 kz = 0;
  if (fabs (normal[0]) > fabs (normal[1]))
  {
    if (fabs (normal[0]) > fabs (normal[2]))
      kz = 0;
    else
      kz = 2;
  }
  else
  {
    if (fabs (normal[1]) > fabs (normal[2]))
      kz = 1;
    else
      kz = 2;
  }
  // nonNormal will be the axis vector most aligned with normal. (future: least aligned?)
  Vec<Float, 3> notNormal;
  notNormal[0] = 0.f;
  notNormal[1] = 0.f;
  notNormal[2] = 0.f;
  notNormal[(kz + 1) % 3] = 1.f; //[M.A.I. 5/31]

  xAxis = cross (normal, notNormal);
  xAxis.normalize ();
  yAxis = cross (normal, xAxis);
  yAxis.normalize ();
}

} // namespace dray
