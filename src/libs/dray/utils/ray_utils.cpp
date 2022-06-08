#include <dray/utils/ray_utils.hpp>
#include <dray/policies.hpp>
#include <dray/transform_3d.hpp>

#include <assert.h>

namespace dray
{

//Array<float32>
//get_gl_depth_buffer(const Array<Ray> &rays,
//                    const Camera &camera,
//                    const float32 near,
//                    const float32 far)
//{
//  return detail::get_gl_depth_buffer(rays,camera, near, far);
//}

//Array<float32>
//get_depth_buffer_img(const Array<Ray> &rays,
//                     const int width,
//                     const int height)
//{
//  Float minv = 1000000.f;
//  Float maxv = -1000000.f;
//
//  int32 size = rays.size();
//  int32 image_size = width * height;
//
//  const Ray *ray_ptr = rays.get_host_ptr_const();
//
//  for(int32 i = 0; i < size;++i)
//  {
//    if(ray_ptr[i].m_near < ray_ptr[i].m_far && ray_ptr[i].m_dist < ray_ptr[i].m_far)
//    {
//      Float depth = ray_ptr[i].m_dist;
//      minv = fminf(minv, depth);
//      maxv = fmaxf(maxv, depth);
//    }
//  }
//
//  Array<float32> dbuffer;
//  dbuffer.resize(image_size* 4);
//  array_memset_zero(dbuffer);
//
//  float32 *d_ptr = dbuffer.get_host_ptr();
//  for (int32 i = 0; i < image_size; i++)
//  {
//    d_ptr[i + 0] = 0.0f;
//    d_ptr[i + 1] = 0.0f;
//    d_ptr[i + 2] = 0.0f;
//    d_ptr[i + 3] = 1.0f;
//  }
//
//
//  float32 len = maxv - minv;
//
//  for(int32 i = 0; i < size;++i)
//  {
//    int32 offset = ray_ptr[i].m_pixel_id * 4;
//    float32 val = 0;
//    if(ray_ptr[i].m_near < ray_ptr[i].m_far && ray_ptr[i].m_dist < ray_ptr[i].m_far)
//    {
//      val = (ray_ptr[i].m_dist - minv) / len;
//    }
//    d_ptr[offset + 0] = val;
//    d_ptr[offset + 1] = val;
//    d_ptr[offset + 2] = val;
//    d_ptr[offset + 3] = 1.f;
//  }
//
//  return dbuffer;
//}

//void save_depth(const Array<Ray> &rays,
//                const int width,
//                const int height,
//                std::string file_name)
//{
//
//  Array<float32> dbuffer = get_depth_buffer_img(rays, width, height);
//  float32 *d_ptr = dbuffer.get_host_ptr();
//
//  PNGEncoder encoder;
//  encoder.encode(d_ptr, width, height);
//  encoder.save(file_name + ".png");
//}

//void save_hitrate(const Array<Ray> &rays, const int32 num_samples, const int width, const int height)
//{
//
//  int32 size = rays.size();
//  int32 image_size = width * height;
//
//  // Read-only host pointers to input ray fields.
//  const int32 *hit_ptr = rays.m_hit_idx.get_host_ptr_const();
//  const int32 *pid_ptr = rays.m_pixel_id.get_host_ptr_const();
//
//  // Result array where we store the normalized count of # hits per bundle.
//  // Values should be between 0 and 1.
//  Array<float32> img_buffer;
//  img_buffer.resize(image_size* 4);
//  float32 *img_ptr = img_buffer.get_host_ptr();
//
//  for (int32 px_channel_idx = 0; px_channel_idx < img_buffer.size(); px_channel_idx++)
//  {
//    //img_ptr[px_channel_idx] = 0;
//    img_ptr[px_channel_idx] = 1;
//  }
//
//  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size / num_samples), [=] DRAY_LAMBDA (int32 bundle_idx)
//  {
//    const int32 b_offset = bundle_idx * num_samples;
//
//    int32 num_hits = 0;
//    for (int32 sample_idx = 0; sample_idx < num_samples; ++sample_idx)
//    {
//      num_hits += (hit_ptr[b_offset + sample_idx] != -1);
//    }
//
//    const float32 hitrate = num_hits / (float32) num_samples;
//
//    const int32 pixel_offset = pid_ptr[b_offset] * 4;
//    img_ptr[pixel_offset + 0] = 1.f - hitrate;
//    img_ptr[pixel_offset + 1] = 1.f - hitrate;
//    img_ptr[pixel_offset + 2] = 1.f - hitrate;
//    img_ptr[pixel_offset + 3] = 1.f;
//  });
//
//  PNGEncoder encoder;
//  encoder.encode(img_ptr, width, height);
//  encoder.save("hitrate.png");
//}

} // namespace dray
