// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ALL_HIT_HPP
#define DRAY_ALL_HIT_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array.hpp>
#include <dray/ray_hit.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/aabb.hpp>

namespace dray
{

DRAY_EXEC_ONLY
bool intersect_AABB_dist(const Vec<float32,4> *bvh,
                         const int32 &currentNode,
                         const Vec<Float,3> &orig_dir,
                         const Vec<Float,3> &inv_dir,
                         const Float& closest_dist,
                         bool &hit_left,
                         bool &hit_right,
                         Float &ldist,
                         Float &rdist,
                         const Float &min_dist) //Find hit after this distance
{
  Vec<float32, 4> first4  = const_get_vec4f(&bvh[currentNode + 0]);
  Vec<float32, 4> second4 = const_get_vec4f(&bvh[currentNode + 1]);
  Vec<float32, 4> third4  = const_get_vec4f(&bvh[currentNode + 2]);
  Float xmin0 = first4[0] * inv_dir[0] - orig_dir[0];
  Float ymin0 = first4[1] * inv_dir[1] - orig_dir[1];
  Float zmin0 = first4[2] * inv_dir[2] - orig_dir[2];
  Float xmax0 = first4[3] * inv_dir[0] - orig_dir[0];
  Float ymax0 = second4[0] * inv_dir[1] - orig_dir[1];
  Float zmax0 = second4[1] * inv_dir[2] - orig_dir[2];
  Float min0 = fmaxf(
    fmaxf(fmaxf(fminf(ymin0, ymax0), fminf(xmin0, xmax0)), fminf(zmin0, zmax0)),
    min_dist);
  Float max0 = fminf(
    fminf(fminf(fmaxf(ymin0, ymax0), fmaxf(xmin0, xmax0)), fmaxf(zmin0, zmax0)),
    closest_dist);
  hit_left = (max0 >= min0);

  Float xmin1 = second4[2] * inv_dir[0] - orig_dir[0];
  Float ymin1 = second4[3] * inv_dir[1] - orig_dir[1];
  Float zmin1 = third4[0] * inv_dir[2] - orig_dir[2];
  Float xmax1 = third4[1] * inv_dir[0] - orig_dir[0];
  Float ymax1 = third4[2] * inv_dir[1] - orig_dir[1];
  Float zmax1 = third4[3] * inv_dir[2] - orig_dir[2];

  Float min1 = fmaxf(
    fmaxf(fmaxf(fminf(ymin1, ymax1), fminf(xmin1, xmax1)), fminf(zmin1, zmax1)),
    min_dist);
  Float max1 = fminf(
    fminf(fminf(fmaxf(ymin1, ymax1), fmaxf(xmin1, xmax1)), fmaxf(zmin1, zmax1)),
    closest_dist);
  hit_right = (max1 >= min1);

  ldist = min0;
  rdist = min1;

  return (min0 > min1);
}

template<class ElemT>
struct FaceIntersector
{
  DeviceMesh<ElemT> m_device_mesh;

  FaceIntersector(DeviceMesh<ElemT> &device_mesh)
   : m_device_mesh(device_mesh)
  {}

  DRAY_EXEC RayHit intersect_face(const Ray &ray,
                                  const int32 &el_idx,
                                  const AABB<2> &ref_box,
                                  stats::Stats &mstat) const
  {
    const bool use_init_guess = true;
    RayHit hit;
    hit.m_hit_idx = -1;

    mstat.acc_candidates(1);
    Vec<Float,2> ref = ref_box.template center<Float> ();
    hit.m_dist = ray.m_near;

    bool inside = Intersector_RayFace<ElemT>::intersect_local (mstat,
                                              m_device_mesh.get_elem(el_idx),
                                              ray,
                                              ref,// initial ref guess
                                              hit.m_dist,  // initial ray guess
                                              use_init_guess);
    if(inside)
    {
      hit.m_hit_idx = el_idx;
      hit.m_ref_pt[0] = ref[0];
      hit.m_ref_pt[1] = ref[1];
    }
    return hit;
  }

};

template <int MAX_DEPTH>
struct DistQ
{
  int32 m_nodes[MAX_DEPTH];
  Float m_distances[MAX_DEPTH];
  int32 m_index;

  DRAY_EXEC void init()
  {
    m_index = -1;
  }

  DRAY_EXEC bool empty()
  {
    return m_index == -1;
  }

  DRAY_EXEC Float peek()
  {
    // could return inf if empty
    return m_distances[m_index];
  }

  DRAY_EXEC Float pop(int32 &node)
  {
    node = m_nodes[m_index];
    Float dist = m_distances[m_index];
    m_index--;
    return dist;
  }

  DRAY_EXEC void insert(const int32 &node, const Float &dist)
  {
    // im sure there are better ways to do this, but i want
    // simple code -ml
    m_index += 1;
    m_nodes[m_index] = node;
    m_distances[m_index] = dist;

    for(int32 i = m_index; i > 0; --i)
    {
      if(m_distances[i] > m_distances[i-1])
      {
        Float tmp_f = m_distances[i];
        int32 tmp_i = m_nodes[i];

        m_distances[i] = m_distances[i-1];
        m_nodes[i] = m_nodes[i-1];

        m_distances[i-1] = tmp_f;
        m_nodes[i-1] = tmp_i;
      }
      else break;
    }
  }
};

template <typename ElemT>
Array<RayHit> intersect_faces_b(Array<Ray> rays, Mesh<ElemT> &mesh)
{
  const int32 size = rays.size();
  Array<RayHit> hits;
  hits.resize(size);
  const BVH bvh = mesh.get_bvh();

  const Ray *ray_ptr = rays.get_device_ptr_const();
  RayHit *hit_ptr = hits.get_device_ptr();

  const int32 *leaf_ptr = bvh.m_leaf_nodes.get_device_ptr_const();
  const int32 *aabb_ids_ptr = bvh.m_aabb_ids.get_device_ptr_const();
  const Vec<float32, 4> *inner_ptr = bvh.m_inner_nodes.get_device_ptr_const();
  const AABB<2> *ref_aabb_ptr = mesh.get_ref_aabbs().get_device_ptr_const();

  DeviceMesh<ElemT> device_mesh(mesh);
  FaceIntersector<ElemT> intersector(device_mesh);

  Array<stats::Stats> mstats;
  mstats.resize(size);
  stats::Stats *mstats_ptr = mstats.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray ray = ray_ptr[i];

    RayHit hit;
    hit.m_hit_idx = -1;

    stats::Stats mstat;
    mstat.construct();

    Float closest_dist = ray.m_far;
    Float min_dist = ray.m_near;
    const Vec<Float,3> dir = ray.m_dir;
    Vec<Float,3> inv_dir;
    inv_dir[0] = rcp_safe(dir[0]);
    inv_dir[1] = rcp_safe(dir[1]);
    inv_dir[2] = rcp_safe(dir[2]);

    int32 current_node;
    DistQ<64> ptodo;
    ptodo.init();
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    ptodo.insert(barrier, infinity<Float>());
    Float current_distance = infinity<Float>();

    const Vec<Float,3> orig = ray.m_orig;

    Vec<Float,3> orig_dir;
    orig_dir[0] = orig[0] * inv_dir[0];
    orig_dir[1] = orig[1] * inv_dir[1];
    orig_dir[2] = orig[2] * inv_dir[2];


    int counter = 0;
    while (current_node != barrier)
    {
      if(current_distance > closest_dist)
      {
        counter++;
      }
      if(ray.m_pixel_id == 477638)
      {
        std::cout<<"current node "<<current_node<<" current_dist "<<current_distance<<" close "<<closest_dist<<"\n";
      }

      if (current_node > -1)
      {
        bool hit_left, hit_right;
        Float ldist, rdist;
        bool right_closer = intersect_AABB_dist(inner_ptr,
                                                current_node,
                                                orig_dir,
                                                inv_dir,
                                                closest_dist,
                                                hit_left,
                                                hit_right,
                                                ldist,
                                                rdist,
                                                min_dist);

        if (!hit_left && !hit_right)
        {
          current_distance = ptodo.pop(current_node);
        }
        else
        {
          Vec<float32, 4> children = const_get_vec4f(&inner_ptr[current_node + 3]);
          int32 l_child;
          constexpr int32 isize = sizeof(int32);
          memcpy(&l_child, &children[0], isize);
          int32 r_child;
          memcpy(&r_child, &children[1], isize);
          current_node = (hit_left) ? l_child : r_child;
          if(hit_left)
          {
            ptodo.insert(l_child, ldist);
          }
          if(hit_right)
          {
            ptodo.insert(r_child, rdist);
          }
          current_distance = ptodo.pop(current_node);
        }
      } // if inner node

      if (current_node < 0 && current_node != barrier) //check register usage
      {
        //if(current_distance > closest_dist) std::cout<<"B";
        current_node = -current_node - 1; //swap the neg address

        int32 el_idx = leaf_ptr[current_node];
        const AABB<2> ref_box = ref_aabb_ptr[aabb_ids_ptr[current_node]];

        RayHit el_hit = intersector.intersect_face(ray, el_idx, ref_box, mstat);

        if(el_hit.m_hit_idx != -1 && el_hit.m_dist < closest_dist && el_hit.m_dist > min_dist)
        {
          hit = el_hit;
          closest_dist = hit.m_dist;
          mstat.found();
        }

        current_distance = ptodo.pop(current_node);
      } // if leaf node

    } //while

    mstats_ptr[i] = mstat;
    hit_ptr[i] = hit;
    if(counter > 10) std::cout<<"id "<<ray.m_pixel_id<<" count "<<counter<<"\n";
  });
  DRAY_ERROR_CHECK();

  stats::StatStore::add_ray_stats(rays, mstats);
  return hits;
}

} // namespace dray
#endif
