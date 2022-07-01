#ifndef DRAY_RAY_UTILS_HPP
#define DRAY_RAY_UTILS_HPP

#include <dray/utils/png_encoder.hpp>
#include <dray/types.hpp>
#include <dray/ray.hpp>
#include <dray/rendering/framebuffer.hpp>

#include <dray/array_utils.hpp>
#include <dray/vec.hpp>   //don't really need.
#include <dray/aabb.hpp>

#include <string>

namespace dray
{

DRAY_EXEC
bool intersect_ray_aabb(const Ray &ray, const AABB<3> &aabb)
{
  const Vec<Float,3> dir_rcp = {rcp_safe(ray.m_dir[0]), rcp_safe(ray.m_dir[1]), rcp_safe(ray.m_dir[2])};
  const Range (&aabbr)[3] = aabb.m_ranges;

  const Float xmin = (aabbr[0].min() - ray.m_orig[0]) * dir_rcp[0];
  const Float ymin = (aabbr[1].min() - ray.m_orig[1]) * dir_rcp[1];
  const Float zmin = (aabbr[2].min() - ray.m_orig[2]) * dir_rcp[2];
  const Float xmax = (aabbr[0].max() - ray.m_orig[0]) * dir_rcp[0];
  const Float ymax = (aabbr[1].max() - ray.m_orig[1]) * dir_rcp[1];
  const Float zmax = (aabbr[2].max() - ray.m_orig[2]) * dir_rcp[2];

  Float left  = fmaxf(fmaxf( fminf(xmin,xmax), fminf(ymin,ymax)), fminf(zmin,zmax));
  Float right = fminf(fminf( fmaxf(xmin,xmax), fmaxf(ymin,ymax)), fmaxf(zmin,zmax));

  return left <= right;
}

// a single channel image of the depth buffer
//Array<float32>
//get_gl_depth_buffer(const Array<Ray> &rays,
//                    const Array<RayHits> &hits,
//                    const Camera &camera,
//                    const float32 near,
//                    const float32 far);
//
//// a grey scale image of the depth buffer
//Array<float32>
//get_depth_buffer_img(const Array<Ray> &rays,
//                     const int width,
//                     const int height);
//
void save_depth(const Array<Ray> &rays,
                const int width,
                const int height,
                std::string file_name = "depth");

/**
 * This function assumes that rays are grouped into bundles each of size (num_samples),
 * and each bundle belongs to the same pixel_id. For each pixel, we count the number of rays
 * which have hit something, and divide the total by (num_samples).
 * The result is a scalar value for each pixel. We output this result to a png image.
 */
//void save_hitrate(const Array<Ray> &rays, const int32 num_samples, const int width, const int height);

}
#endif


