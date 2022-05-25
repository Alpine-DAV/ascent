// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/billboard.hpp>
#include <dray/rendering/device_framebuffer.hpp>
#include <dray/rendering/font_factory.hpp>
#include <dray/array_utils.hpp>
#include <dray/error.hpp>
#include <dray/matrix.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/rendering/screen_text_annotator.hpp>
#include <dray/rendering/colors.hpp>

namespace dray
{

namespace detail
{

AABB<3> bound_sphere(const Vec<float32,3> &center, const float32 radius)
{
  AABB<3> res;
  Vec<float32,3> temp;

  temp[0] = radius;
  temp[1] = 0.f;
  temp[2] = 0.f;

  res.include(center + temp);
  res.include(center - temp);

  temp[0] = 0.f;
  temp[1] = radius;
  temp[2] = 0.f;

  res.include(center + temp);
  res.include(center - temp);

  temp[0] = 0.f;
  temp[1] = 0.f;
  temp[2] = radius;

  res.include(center + temp);
  res.include(center - temp);

  return res;
}
// I need to consolidate all this
template <typename T>
DRAY_EXEC_ONLY bool intersect_AABB (const Vec<float32, 4> *bvh,
                                    const int32 &currentNode,
                                    const Vec<T, 3> &orig_dir,
                                    const Vec<T, 3> &inv_dir,
                                    const T &closest_dist,
                                    bool &hit_left,
                                    bool &hit_right,
                                    const T &min_dist) // Find hit after this distance
{
  Vec<float32, 4> first4 = const_get_vec4f (&bvh[currentNode + 0]);
  Vec<float32, 4> second4 = const_get_vec4f (&bvh[currentNode + 1]);
  Vec<float32, 4> third4 = const_get_vec4f (&bvh[currentNode + 2]);
  T xmin0 = first4[0] * inv_dir[0] - orig_dir[0];
  T ymin0 = first4[1] * inv_dir[1] - orig_dir[1];
  T zmin0 = first4[2] * inv_dir[2] - orig_dir[2];
  T xmax0 = first4[3] * inv_dir[0] - orig_dir[0];
  T ymax0 = second4[0] * inv_dir[1] - orig_dir[1];
  T zmax0 = second4[1] * inv_dir[2] - orig_dir[2];
  T min0 =
  fmaxf (fmaxf (fmaxf (fminf (ymin0, ymax0), fminf (xmin0, xmax0)), fminf (zmin0, zmax0)),
         min_dist);
  T max0 =
  fminf (fminf (fminf (fmaxf (ymin0, ymax0), fmaxf (xmin0, xmax0)), fmaxf (zmin0, zmax0)),
         closest_dist);
  hit_left = (max0 >= min0);

  T xmin1 = second4[2] * inv_dir[0] - orig_dir[0];
  T ymin1 = second4[3] * inv_dir[1] - orig_dir[1];
  T zmin1 = third4[0] * inv_dir[2] - orig_dir[2];
  T xmax1 = third4[1] * inv_dir[0] - orig_dir[0];
  T ymax1 = third4[2] * inv_dir[1] - orig_dir[1];
  T zmax1 = third4[3] * inv_dir[2] - orig_dir[2];

  T min1 =
  fmaxf (fmaxf (fmaxf (fminf (ymin1, ymax1), fminf (xmin1, xmax1)), fminf (zmin1, zmax1)),
         min_dist);
  T max1 =
  fminf (fminf (fminf (fmaxf (ymin1, ymax1), fmaxf (xmin1, xmax1)), fmaxf (zmin1, zmax1)),
         closest_dist);
  hit_right = (max1 >= min1);

  return (min0 > min1);
}

}// namespace detail

struct DeviceBillboard
{
  const float32 *m_texture;
  const int32 m_texture_width;
  const int32 m_texture_height;
  const Vec<Float,3> m_ray_diff_x;
  const Vec<Float,3> m_ray_diff_y;
  const Vec<float32,3> *m_anchors;
  const Vec<float32,2> *m_offsets;
  const Vec<float32,2> *m_dims;
  const Vec<float32,2> *m_tcoords;
  const Vec<float32,3> m_up;
  const Vec<float32,3> m_text_color;

  DeviceBillboard() = delete;
  DeviceBillboard(Billboard &b)
    : m_texture(b.m_texture.get_device_ptr_const()),
      m_texture_width(b.m_texture_width),
      m_texture_height(b.m_texture_height),
      m_ray_diff_x(b.m_ray_differential_x),
      m_ray_diff_y(b.m_ray_differential_y),
      m_anchors(b.m_anchors.get_device_ptr_const()),
      m_offsets(b.m_offsets.get_device_ptr_const()),
      m_dims(b.m_dims.get_device_ptr_const()),
      m_tcoords(b.m_tcoords.get_device_ptr_const()),
      m_up(b.m_up),
      m_text_color(b.m_text_color)
  {}

  DRAY_EXEC
  RayHit intersect_billboard(const Ray &ray,
                             const int32 index) const
  {
    const Vec<float32,3> anchor = m_anchors[index];
    const Vec<float32,2> dims = m_dims[index];
    //Ray tracing Gems II billboard intersections

    // ray will be either float64 or float32
    // but out bill board math is all float32

    Vec<Float,3> ray_normal_dir = ray.m_orig - anchor;
    Vec<float32,3> normal_dir;
    type_convert(ray_normal_dir,normal_dir);

    bool y_align = false;
    Vec<float32,3> n = normal_dir;
    if(y_align)
    {
      n = {{normal_dir[0], 0.f, normal_dir[2]}};
    }

    n.normalize();
    Vec<float32,3> t = cross(m_up, n);
    t.normalize();
    Vec<float32,3> b = cross(n, t);



    Matrix<float32,3,3> to_tangent;
    to_tangent.set_col(0,t);
    to_tangent.set_col(1,b);
    to_tangent.set_col(2,n);
    to_tangent = to_tangent.transpose();

    // again float64 vs float32 issue
    Vec<float32,3> dir;
    type_convert(ray.m_dir,dir);

    Vec<float32,3> ray_dp = to_tangent * dir;
    Vec<float32,3> ray_op = to_tangent * normal_dir;

    Vec<float32,3> dp;
    Vec<float32,3> op;
    type_convert(ray_dp,dp);
    type_convert(ray_op,op);

    float32 s = -op[2] / dp[2];
    Vec<float32,2> pp; // point on billboard plane
    pp[0] = op[0] + s * dp[0];
    pp[1] = op[1] + s * dp[1];

    // check if the plane coordinates are within the dims
    RayHit hit;
    hit.init();

    // the anchor can be associated with any position in the billboard
    // the offset defines the reference space coords [0,1] so we need to
    // add the offset to the plane position and check for inside
    // the billboard
    const Vec<float32,2> offset = m_offsets[index];
    pp[0] += dims[0] * offset[0];
    pp[1] += dims[1] * offset[1];
    //if((abs(pp[0]) < 0.5f * dims[0]) && (abs(pp[1]) < 0.5f * dims[1]))

    if(pp[0] > 0.f && pp[0] < dims[0] && pp[1] < dims[1] && pp[1] > 0.f)
    {
      //hit.m_ref_pt[0] = pp[0] / dims[0] + 0.5f;
      //hit.m_ref_pt[1] = pp[1] / dims[1] + 0.5f;
      hit.m_ref_pt[0] = pp[0] / dims[0];
      hit.m_ref_pt[1] = pp[1] / dims[1];
      hit.m_dist = s;
      hit.m_hit_idx = index;
    }
    return hit;
  }

  DRAY_EXEC
  Vec<float32,2> lerp_tbox(const AABB<2> &tbox, const Vec<float32,2> uv) const
  {
    float32 s = lerp(tbox.m_ranges[0].min(),tbox.m_ranges[0].max(), uv[0]);
    float32 t = lerp(tbox.m_ranges[1].min(),tbox.m_ranges[1].max(), uv[1]);
    Vec<float32,2> st = {{s,t}};
    return st;
  }

  DRAY_EXEC
  void texture_derivs(const Ray &ray,
                      const RayHit &hit,
                      const AABB<2> &tbox,
                      const Vec<float32,2> &st,
                      Vec<float32,2> &st_x,
                      Vec<float32,2> &st_y) const
  {
    Ray ray_x = ray;
    Ray ray_y = ray;
    ray_x.m_dir = ray.m_dir + m_ray_diff_x;
    ray_y.m_dir = ray.m_dir + m_ray_diff_y;
    // this will load the centers/dims twice so
    // maybe its better to load the dims first
    RayHit hit_x = intersect_billboard(ray_x, hit.m_hit_idx);
    RayHit hit_y = intersect_billboard(ray_y, hit.m_hit_idx);

    st_x = st;
    if(hit_x.m_dist != infinity<float32>())
    {
      Vec<float32,2> uv_x = {{ (float32)hit_x.m_ref_pt[0], (float32)hit_x.m_ref_pt[1]}};
      st_x = lerp_tbox(tbox, uv_x);
      st_x[0] *= m_texture_width;
      st_x[1] *= m_texture_height;
    }

    st_y = st;
    if(hit_y.m_dist != infinity<float32>())
    {
      Vec<float32,2> uv_y = {{ (float32)hit_y.m_ref_pt[0], (float32)hit_y.m_ref_pt[1]}};
      st_y = lerp_tbox(tbox, uv_y);
      st_y[0] *= m_texture_width;
      st_y[1] *= m_texture_height;
    }

  }

  DRAY_EXEC
  float32 tblerp(const float32 s, const float32 t) const
  {
    // we now need to blerp
    Vec<int32,2> st_min, st_max;
    st_min[0] = clamp(int32(s), 0, m_texture_width - 1);
    st_min[1] = clamp(int32(t), 0, m_texture_height - 1);
    st_max[0] = clamp(st_min[0]+1, 0, m_texture_width - 1);
    st_max[1] = clamp(st_min[1]+1, 0, m_texture_height - 1);

    Vec<float32,4> vals;
    vals[0] = m_texture[st_min[1] * m_texture_width + st_min[0]];
    vals[1] = m_texture[st_min[1] * m_texture_width + st_max[0]];
    vals[2] = m_texture[st_max[1] * m_texture_width + st_min[0]];
    vals[3] = m_texture[st_max[1] * m_texture_width + st_max[0]];

    float32 dx = s - float32(st_min[0]);
    float32 dy = t - float32(st_min[1]);

    float32 x0 = lerp(vals[0], vals[1], dx);
    float32 x1 = lerp(vals[2], vals[3], dx);
    // this the signed distance to the glyph
    return lerp(x0, x1, dy);
  }
  
  DRAY_EXEC
  float32 alpha(const Ray &ray, const RayHit &hit) const
  {
    if(hit.m_hit_idx == -1)
    {
      return 0.f;
    }

    const int32 offset = hit.m_hit_idx * 4;
    AABB<2> tbox;
    tbox.include(m_tcoords[offset + 0]); // bl
    tbox.include(m_tcoords[offset + 1]); // br
    tbox.include(m_tcoords[offset + 2]); // tl
    tbox.include(m_tcoords[offset + 3]); // tr

    Vec<float32,2> uv = {{(float32)hit.m_ref_pt[0], (float32)hit.m_ref_pt[1]}};

    Vec<float32,2> st = lerp_tbox(tbox, uv);
    st[0] *= m_texture_width;
    st[1] *= m_texture_height;

    float32 distance = this->tblerp(st[0], st[1]);
    Vec<float32,2> st_x, st_y;
    texture_derivs(ray, hit, tbox, st, st_x, st_y);

    float32 d_x1 = tblerp(st_x[0],st_x[1]);
    float32 d_y1 = tblerp(st_y[0], st_y[1]);

    // forward difference of the distance value
    float32 dfx = d_x1 - distance;
    float32 dfy = d_y1 - distance;
    float32 width = 0.7f * sqrt(dfx*dfx + dfy*dfy);
    float32 res = smoothstep(0.5f-width,0.5f+width,distance);

    // super sample
    constexpr float32 dscale = 0.354f;
    Vec<float32,2> ss_dy = dscale * (st_y - st);
    Vec<float32,2> ss_dx = dscale * (st_x - st);

    Vec<float32,2> ss = st - ss_dy - ss_dx;
    float32 ss_0 = tblerp(ss[0], ss[1]);
    ss = st - ss_dy + ss_dx;
    float32 ss_1 = tblerp(ss[0], ss[1]);
    ss = st + ss_dy - ss_dx;
    float32 ss_2 = tblerp(ss[0], ss[1]);
    ss = st + ss_dy + ss_dx;
    float32 ss_3 = tblerp(ss[0], ss[1]);
    ss_0 = smoothstep(0.5f-width,0.5f+width,ss_0);
    ss_1 = smoothstep(0.5f-width,0.5f+width,ss_1);
    ss_2 = smoothstep(0.5f-width,0.5f+width,ss_2);
    ss_3 = smoothstep(0.5f-width,0.5f+width,ss_3);
    res= (res + 0.5f * (ss_0 + ss_1 + ss_2 + ss_3)) / 3.f;

    return res;
  }

};

Billboard::Billboard(const std::vector<std::string> &texts,
                     const std::vector<Vec<float32,3>> &anchors,
                     const std::vector<Vec<float32,2>> &offsets,
                     const std::vector<float32> &world_sizes)
  : m_up({0.f, 1.f, 0.f}),
    m_ray_differential_x({0.f, 0.f, 0.f}),
    m_ray_differential_y({0.f, 0.f, 0.f}),
    m_text_color({0.f, 0.f, 0.f})
{
  Font *font = FontFactory::font("OpenSans-Regular");
  ScreenTextAnnotator anot;

  Array<float32> texture;
  int32 twidth,theight;
  Array<AABB<2>> tboxs, pboxs;
  anot.render_to_texture(texts, texture, twidth, theight, tboxs, pboxs);
  // world size of the font
  const int32 size = texts.size();

  Array<AABB<3>> aabbs;

  m_anchors.resize(size);
  m_offsets.resize(size);
  m_dims.resize(size);
  m_tcoords.resize(size*4); // this is a quad
  aabbs.resize(size);

  Vec<float32,3> *anchor_ptr = m_anchors.get_host_ptr();
  Vec<float32,2> *offset_ptr = m_offsets.get_host_ptr();
  Vec<float32,2> *dims_ptr = m_dims.get_host_ptr();
  Vec<float32,2> *tcoords_ptr = m_tcoords.get_host_ptr();
  AABB<3> *aabbs_ptr = aabbs.get_host_ptr();
  const AABB<2> *pbox_ptr = pboxs.get_host_ptr_const();
  const AABB<2> *tbox_ptr = tboxs.get_host_ptr_const();


  for(int i = 0; i < size; ++i)
  {
    // parametric coords of where the anchor is [0,1]
    // (0.5, 0.5) would place the anchor at the center of the box
    // (0.0, 0.0) would place the anchor at the bottom left of the box
    Vec<float32,2> offset = offsets[i];
    if(offset[0] < 0.f || offset[0] > 1.f ||
       offset[1] < 0.f || offset[1] > 1.f)
    {
      DRAY_ERROR("Billboard offsets must be in the range [0,1]");
    }

    Vec<float32,3> anchor = anchors[i];
    Vec<float32,2> width_height;
    width_height[0] = pbox_ptr[i].m_ranges[0].length() * world_sizes[i];
    width_height[1] = pbox_ptr[i].m_ranges[1].length() * world_sizes[i];
    // To construct the bounding box for the BVH, just take the max dim and
    // use that as the radius of a sphere.
    // Note: since the billboard can swing any way the camera is pointing,
    //       this is a less than optimal bounding box, especially if the
    //       anchor is on the edge of the billboard. There shouldn't be too
    //       many so that is likely not an issue. The alternative would be to
    //       create a new bvh/ or add the ability to udpate the bvh each time.
    Vec<float32,2> rot_pt = {{width_height[0] * offset[0],
                              width_height[1] * offset[1]}};
    float distance = 0.f;
    const Vec<float32,2> corners[4] = {
                                        {{            0.f, 0.f}},
                                        {{            0.f, width_height[0]}},
                                        {{width_height[1], 0.f}},
                                        {{width_height[1], width_height[0]}},
                                      };
    // find the furthest corner
    for(int c = 0; c < 4; c++)
    {
      distance = std::max(distance, (rot_pt - corners[c]).magnitude());
    }

    aabbs_ptr[i] = detail::bound_sphere(anchor, distance);
    /*
    std::cout<<"Text "<<texts[i]<<"\n";
    std::cout<<"   anchor : "<<anchor<<"\n";
    std::cout<<"   dims : "<<width_height<<"\n";
    std::cout<<"   offset : "<<offset<<"\n\n";
    */
    anchor_ptr[i] = anchor;
    offset_ptr[i] = offset;
    dims_ptr[i] = width_height;

    AABB<2> t_box = tbox_ptr[i];

    Vec<float32,2> bottom_left;
    bottom_left[0] = t_box.m_ranges[0].min();
    bottom_left[1] = t_box.m_ranges[1].min();

    Vec<float32,2> bottom_right;
    bottom_right[0] = t_box.m_ranges[0].max();
    bottom_right[1] = t_box.m_ranges[1].min();

    Vec<float32,2> top_left;
    top_left[0] = t_box.m_ranges[0].min();
    top_left[1] = t_box.m_ranges[1].max();

    Vec<float32,2> top_right;
    top_right[0] = t_box.m_ranges[0].max();
    top_right[1] = t_box.m_ranges[1].max();

    const int32 toffset = i * 4;
    tcoords_ptr[toffset + 0] = bottom_left;
    tcoords_ptr[toffset + 1] = bottom_right;
    tcoords_ptr[toffset + 2] = top_left;
    tcoords_ptr[toffset + 3] = top_right;
  }

  LinearBVHBuilder builder;
  m_bvh = builder.construct (aabbs);
  m_texture = texture;
  m_texture_width = twidth;
  m_texture_height = theight;
}

void Billboard::save_texture(const std::string file_name)
{
  // write out texture for debugging
  Array<float32> timage;
  const int32 size = m_texture_width * m_texture_height;
  timage.resize(size * 4);
  float32 *image_ptr = timage.get_host_ptr();
  for(int32 i = 0; i < size; ++i)
  {
    const int32 offset = i * 4;
    const float32 val = m_texture.get_value(i);
    image_ptr[offset + 0] = val;
    image_ptr[offset + 1] = val;
    image_ptr[offset + 2] = val;
    image_ptr[offset + 3] = 1.f;
  }

  PNGEncoder encoder;
  encoder.encode(image_ptr, m_texture_width, m_texture_height);
  encoder.save(file_name + ".png");
}

void Billboard::shade(const Array<Ray> &rays, const Array<RayHit> &hits, Framebuffer &fb)
{
  DeviceFramebuffer d_framebuffer(fb);

  const RayHit *hit_ptr = hits.get_device_ptr_const();
  const Ray *rays_ptr = rays.get_device_ptr_const();
  const Vec<float32,3> text_color = m_text_color;

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, hits.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    if (hit.m_hit_idx != -1)
    {
      const int32 pixel_id = rays_ptr[ii].m_pixel_id;

      Vec<float32,4> fcolor = {{ text_color[0],
                                 text_color[1],
                                 text_color[2],
                                 (float32)hit.m_ref_pt[2]}};

      Vec<float32,4> color = d_framebuffer.m_colors[pixel_id];
      blend_pre_alpha(fcolor, color);
      d_framebuffer.m_colors[pixel_id] = fcolor;
      d_framebuffer.m_depths[pixel_id] = hit.m_dist;
    }
  });
  DRAY_ERROR_CHECK();
}

AABB<3> Billboard::bounds()
{
  return m_bvh.m_bounds;
}

Array<RayHit> Billboard::intersect (const Array<Ray> &rays)
{
  const int32 *leaf_ptr = m_bvh.m_leaf_nodes.get_device_ptr_const ();
  const Vec<float32, 4> *inner_ptr = m_bvh.m_inner_nodes.get_device_ptr_const ();

  const Ray *ray_ptr = rays.get_device_ptr_const ();

  const int32 size = rays.size ();

  Array<RayHit> hits;
  hits.resize (size);
  const Vec<float32,3> up_dir = m_up;

  RayHit *hit_ptr = hits.get_device_ptr ();
  DeviceBillboard dbillboard(*this);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i)
  {

    Ray ray = ray_ptr[i];

    RayHit hit;
    hit.init();

    float32 closest_dist = ray.m_far;
    float32 min_dist = ray.m_near;
    Vec<float32, 3> dir;
    type_convert(ray.m_dir,dir);
    Vec<float32, 3> inv_dir;
    inv_dir[0] = rcp_safe (dir[0]);
    inv_dir[1] = rcp_safe (dir[1]);
    inv_dir[2] = rcp_safe (dir[2]);

    int32 current_node;
    int32 todo[64];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;

    Vec<float32, 3> orig_dir;
    orig_dir[0] = ray.m_orig[0] * inv_dir[0];
    orig_dir[1] = ray.m_orig[1] * inv_dir[1];
    orig_dir[2] = ray.m_orig[2] * inv_dir[2];

    while (current_node != barrier)
    {
      if (current_node > -1)
      {
        bool hit_left, hit_right;
        bool right_closer = detail::intersect_AABB (inner_ptr,
                                                    current_node,
                                                    orig_dir,
                                                    inv_dir,
                                                    closest_dist,
                                                    hit_left,
                                                    hit_right,
                                                    min_dist);

        if (!hit_left && !hit_right)
        {
          current_node = todo[stackptr];
          stackptr--;
        }
        else
        {
          Vec<float32, 4> children = const_get_vec4f (&inner_ptr[current_node + 3]);
          int32 l_child;
          constexpr int32 isize = sizeof (int32);
          memcpy (&l_child, &children[0], isize);
          int32 r_child;
          memcpy (&r_child, &children[1], isize);
          current_node = (hit_left) ? l_child : r_child;

          if (hit_left && hit_right)
          {
            if (right_closer)
            {
              current_node = r_child;
              stackptr++;
              todo[stackptr] = l_child;
            }
            else
            {
              stackptr++;
              todo[stackptr] = r_child;
            }
          }
        }
      } // if inner node

      if (current_node < 0 && current_node != barrier) // check register usage
      {
        current_node = -current_node - 1; // swap the neg address

        const int32 bill_index = leaf_ptr[current_node];

        RayHit bill_hit = dbillboard.intersect_billboard(ray, bill_index);

        if(bill_hit.m_dist < closest_dist && bill_hit.m_dist > min_dist)
        {
          // check to see if we are transparent or not
          // Note: the best way I can thing of to handle transparency
          // is to use something like a multi-hit query. This will be
          // good enough for now and avoids complications
          float32 alpha = dbillboard.alpha(ray, bill_hit);
          // we already calculate this, so just store it for later
          // use in the shader
          bill_hit.m_ref_pt[2] = alpha;
          if(alpha > 0.05f)
          {
            hit.m_hit_idx = bill_index;
            hit.m_ref_pt = bill_hit.m_ref_pt;
            hit.m_dist = bill_hit.m_dist;
            closest_dist = bill_hit.m_dist;
          }
        }

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } // while

    hit_ptr[i] = hit;
  });
  DRAY_ERROR_CHECK();
  return hits;
}

void Billboard::camera(const Camera& camera)
{
  m_up = camera.get_up();
  m_ray_differential_x = camera.ray_differential_x();
  m_ray_differential_y = camera.ray_differential_y();
}

void Billboard::text_color(const Vec<float32,3> &color)
{
  m_text_color = color;
}

}; //namepspace dray
