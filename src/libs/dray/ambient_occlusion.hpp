// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_AMBIENT_OCCLUSION_HPP
#define DRAY_AMBIENT_OCCLUSION_HPP

#include <dray/intersection_context.hpp>

//#include <dray/aabb.hpp>
#include <dray/ray.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class AmbientOcclusion
{

  public:
  // Not sure where these functions should go...

  const static Float nudge_dist;

  /**
   * [in] intersection_ctx
   * [in] occ_samples
   * [in] occ_near
   * [in] occ_far
   *
   * returns occ_rays
   */
  static Array<Ray> gen_occlusion (const Array<IntersectionContext> intersection_ctx,
                                   const int32 occ_samples,
                                   const Float occ_near,
                                   const Float occ_far);
  static Array<Ray> gen_occlusion (const Array<IntersectionContext> intersection_ctx,
                                   const int32 occ_samples,
                                   const Float occ_near,
                                   const Float occ_far,
                                   Array<int32> &compact_indexing);
  // Note: We return type Ray<T> instead of [out] parameter, because the calling
  // code does not know how many occlusion rays there will be. (It will be a
  // multiple of the number of valid primary intersections, but the calling code
  // does not know how many valid primary intersections there are.)

  // ------------

  // These sampling methods can definitely be moved out of AmbientOcclusion.
  // These sampling methods were adapted from https://gitlab.kitware.com/mclarsen/vtk-m/blob/pathtracer/vtkm/rendering/raytracing/Sampler.h

  // static Vec<T,3> CosineWeightedHemisphere(const int32 &sampleNum);
  // static void ConstructTangentBasis( const Vec<T,3> &normal, Vec<T,3> &xAxis, Vec<T,3> &yAxis);
  DRAY_EXEC static Vec<Float, 3> CosineWeightedHemisphere (const int32 &sampleNum);
  DRAY_EXEC static void ConstructTangentBasis (const Vec<Float, 3> &normal,
                                               Vec<Float, 3> &xAxis,
                                               Vec<Float, 3> &yAxis);
};
} // namespace dray
#endif // DRAY_AMBIENT_OCCLUSION_HPP
