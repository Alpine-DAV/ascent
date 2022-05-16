// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/linear_bvh_builder.hpp>

#include <dray/array_utils.hpp>
#include <dray/error_check.hpp>
#include <dray/math.hpp>
#include <dray/morton_codes.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

namespace dray
{

AABB<> reduce (const Array<AABB<>> &aabbs)
{


  RAJA::ReduceMin<reduce_policy, float32> xmin (infinity32 ());
  RAJA::ReduceMin<reduce_policy, float32> ymin (infinity32 ());
  RAJA::ReduceMin<reduce_policy, float32> zmin (infinity32 ());

  RAJA::ReduceMax<reduce_policy, float32> xmax (neg_infinity32 ());
  RAJA::ReduceMax<reduce_policy, float32> ymax (neg_infinity32 ());
  RAJA::ReduceMax<reduce_policy, float32> zmax (neg_infinity32 ());

  Timer timer;
  const AABB<> *aabb_ptr = aabbs.get_device_ptr_const ();
  DRAY_LOG_ENTRY ("reduce_setup", timer.elapsed ());
  timer.reset ();
  // const AABB<> *aabb_ptr = aabbs.get_host_ptr_const();
  const int size = aabbs.size ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    const AABB<> aabb = aabb_ptr[i];
    // std::cout<<i<<" "<<aabb<<"\n";
    xmin.min (aabb.m_ranges[0].min ());
    ymin.min (aabb.m_ranges[1].min ());
    zmin.min (aabb.m_ranges[2].min ());

    xmax.max (aabb.m_ranges[0].max ());
    ymax.max (aabb.m_ranges[1].max ());
    zmax.max (aabb.m_ranges[2].max ());
  });
  DRAY_ERROR_CHECK();

  AABB<> res;
  Vec3f mins = make_vec3f (xmin.get (), ymin.get (), zmin.get ());
  Vec3f maxs = make_vec3f (xmax.get (), ymax.get (), zmax.get ());

  res.include (mins);
  res.include (maxs);
  return res;
}

Array<uint32> get_mcodes (Array<AABB<>> &aabbs, const AABB<> &bounds)
{
  Vec3f min_coord (bounds.min ());
  Vec3f extent (bounds.max () - bounds.min ());
  Vec3f inv_extent;

  for (int i = 0; i < 3; ++i)
  {
    inv_extent[i] = (extent[i] == .0f) ? 0.f : 1.f / extent[i];
  }

  const int size = aabbs.size ();
  Array<uint32> mcodes;
  mcodes.resize (size);

  const AABB<> *aabb_ptr = aabbs.get_device_ptr_const ();
  uint32 *mcodes_ptr = mcodes.get_device_ptr ();

  // std::cout<<aabbs.get_host_ptr_const()[0]<<"\n";
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    const AABB<> aabb = aabb_ptr[i];
    // get the center and normalize it
    float32 centroid_x = (aabb.m_ranges[0].center () - min_coord[0]) * inv_extent[0];
    float32 centroid_y = (aabb.m_ranges[1].center () - min_coord[1]) * inv_extent[1];
    float32 centroid_z = (aabb.m_ranges[2].center () - min_coord[2]) * inv_extent[2];
    mcodes_ptr[i] = morton_3d (centroid_x, centroid_y, centroid_z);
  });
  DRAY_ERROR_CHECK();

  return mcodes;
}


//
// reorder and array based on a new set of indices.
// array   [a,b,c]
// indices [1,0,2]
// result  [b,a,c]
//
template <typename T> void reorder (Array<int32> &indices, Array<T> &array)
{
  assert (indices.size () == array.size ());
  const int size = array.size ();

  Array<T> temp;
  temp.resize (size);

  T *temp_ptr = temp.get_device_ptr ();
  const T *array_ptr = array.get_device_ptr_const ();
  const int32 *indices_ptr = indices.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    int32 in_idx = indices_ptr[i];
    temp_ptr[i] = array_ptr[in_idx];
  });
  DRAY_ERROR_CHECK();

  array = temp;
}

Array<int32> sort_mcodes (Array<uint32> &mcodes)
{
  const int size = mcodes.size ();
  Array<int32> iter = array_counting (size, 0, 1);
  // TODO: create custom sort for GPU / CPU
  int32 *iter_ptr = iter.get_host_ptr ();
  uint32 *mcodes_ptr = mcodes.get_host_ptr ();

  std::sort (iter_ptr, iter_ptr + size,
             [=] (int32 i1, int32 i2) { return mcodes_ptr[i1] < mcodes_ptr[i2]; });


  reorder (iter, mcodes);

  return iter;
}

struct BVHData
{
  Array<int32> m_left_children;
  Array<int32> m_right_children;
  Array<int32> m_parents;
  Array<int32> m_leafs;
  Array<uint32> m_mcodes;
  Array<AABB<>> m_inner_aabbs;
  Array<AABB<>> m_leaf_aabbs;
};


DRAY_EXEC int32 delta (const int32 &a, const int32 &b, const int32 &inner_size, const uint32 *mcodes)
{
  bool tie = false;
  bool out_of_range = (b < 0 || b > inner_size);
  // still make the call but with a valid adderss
  const int32 bb = (out_of_range) ? 0 : b;
  const uint32 acode = mcodes[a];
  const uint32 bcode = mcodes[bb];
  // use xor to find where they differ
  uint32 exor = acode ^ bcode;
  tie = (exor == 0);
  // break the tie, a and b must always differ
  exor = tie ? uint32 (a) ^ uint32 (bb) : exor;
  int32 count = clz (exor);
  if (tie) count += 32;
  count = (out_of_range) ? -1 : count;
  return count;
}

void build_tree (BVHData &data)
{
  // http://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf
  const int32 inner_size = data.m_left_children.size ();

  int32 *lchildren_ptr = data.m_left_children.get_device_ptr ();
  int32 *rchildren_ptr = data.m_right_children.get_device_ptr ();
  int32 *parent_ptr = data.m_parents.get_device_ptr ();
  const uint32 *mcodes_ptr = data.m_mcodes.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, inner_size), [=] DRAY_LAMBDA (int32 i) {
    // determine range direction
    int32 d = 0 > (delta (i, i + 1, inner_size, mcodes_ptr) -
                   delta (i, i - 1, inner_size, mcodes_ptr)) ?
              -1 :
              1;

    // find upper bound for the length of the range
    int32 min_delta = delta (i, i - d, inner_size, mcodes_ptr);
    int32 lmax = 2;
    while (delta (i, i + lmax * d, inner_size, mcodes_ptr) > min_delta)
      lmax *= 2;

    // binary search to find the lower bound
    int32 l = 0;
    for (int32 t = lmax / 2; t >= 1; t /= 2)
    {
      if (delta (i, i + (l + t) * d, inner_size, mcodes_ptr) > min_delta)
      {
        l += t;
      }
    }

    int32 j = i + l * d;
    int32 delta_node = delta (i, j, inner_size, mcodes_ptr);
    int32 s = 0;
    float32 div_factor = 2.f;
    // find the split postition using a binary search
    for (int32 t = (int32)ceil (float32 (l) / div_factor);;
         div_factor *= 2, t = (int32)ceil (float32 (l) / div_factor))
    {
      if (delta (i, i + (s + t) * d, inner_size, mcodes_ptr) > delta_node)
      {
        s += t;
      }
      if (t == 1) break;
    }

    int32 split = i + s * d + min (d, 0);
    // assign parent/child pointers
    if (min (i, j) == split)
    {
      // leaf
      parent_ptr[split + inner_size] = i;
      lchildren_ptr[i] = split + inner_size;
    }
    else
    {
      // inner node
      parent_ptr[split] = i;
      lchildren_ptr[i] = split;
    }


    if (max (i, j) == split + 1)
    {
      // leaf
      parent_ptr[split + inner_size + 1] = i;
      rchildren_ptr[i] = split + inner_size + 1;
    }
    else
    {
      parent_ptr[split + 1] = i;
      rchildren_ptr[i] = split + 1;
    }

    if (i == 0)
    {
      // flag the root
      parent_ptr[0] = -1;
    }
  });
  DRAY_ERROR_CHECK();
}

void propagate_aabbs (BVHData &data)
{
  const int inner_size = data.m_inner_aabbs.size ();
  const int leaf_size = data.m_leafs.size ();

  Array<int32> counters;
  counters.resize (inner_size);

  array_memset_zero (counters);

  const int32 *lchildren_ptr = data.m_left_children.get_device_ptr_const ();
  const int32 *rchildren_ptr = data.m_right_children.get_device_ptr_const ();
  const int32 *parent_ptr = data.m_parents.get_device_ptr_const ();
  const AABB<> *leaf_aabb_ptr = data.m_leaf_aabbs.get_device_ptr_const ();

  AABB<> *inner_aabb_ptr = data.m_inner_aabbs.get_device_ptr ();
  int32 *counter_ptr = counters.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, leaf_size), [=] DRAY_LAMBDA (int32 i) {
    int32 current_node = parent_ptr[inner_size + i];

    while (current_node != -1)
    {
      int32 old = RAJA::atomicAdd<atomic_policy> (&(counter_ptr[current_node]), 1);

      if (old == 0)
      {
        // first thread to get here kills itself
        return;
      }

      int32 lchild = lchildren_ptr[current_node];
      int32 rchild = rchildren_ptr[current_node];
      // gather the aabbs
      AABB<> aabb;
      if (lchild >= inner_size)
      {
        aabb.include (leaf_aabb_ptr[lchild - inner_size]);
      }
      else
      {
        aabb.include (inner_aabb_ptr[lchild]);
      }

      if (rchild >= inner_size)
      {
        aabb.include (leaf_aabb_ptr[rchild - inner_size]);
      }
      else
      {
        aabb.include (inner_aabb_ptr[rchild]);
      }

      inner_aabb_ptr[current_node] = aabb;

      current_node = parent_ptr[current_node];
    }

    // printf("There can be only one\n");
  });
  DRAY_ERROR_CHECK();

  // AABB<> *inner = data.m_inner_aabbs.get_host_ptr();
  // std::cout<<"Root bounds "<<inner[0]<<"\n";
}

Array<Vec<float32, 4>> emit (BVHData &data)
{
  const int inner_size = data.m_inner_aabbs.size ();

  const int32 *lchildren_ptr = data.m_left_children.get_device_ptr_const ();
  const int32 *rchildren_ptr = data.m_right_children.get_device_ptr_const ();

  const AABB<> *leaf_aabb_ptr = data.m_leaf_aabbs.get_device_ptr_const ();
  const AABB<> *inner_aabb_ptr = data.m_inner_aabbs.get_device_ptr_const ();

  Array<Vec<float32, 4>> flat_bvh;
  flat_bvh.resize (inner_size * 4);

  Vec<float32, 4> *flat_ptr = flat_bvh.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, inner_size), [=] DRAY_LAMBDA (int32 node) {
    Vec<float32, 4> vec1;
    Vec<float32, 4> vec2;
    Vec<float32, 4> vec3;
    Vec<float32, 4> vec4;

    AABB<> l_aabb, r_aabb;

    int32 lchild = lchildren_ptr[node];
    if (lchild >= inner_size)
    {
      l_aabb = leaf_aabb_ptr[lchild - inner_size];
      lchild = -(lchild - inner_size + 1);
    }
    else
    {
      l_aabb = inner_aabb_ptr[lchild];
      // do the offset now
      lchild *= 4;
    }

    int32 rchild = rchildren_ptr[node];
    if (rchild >= inner_size)
    {
      r_aabb = leaf_aabb_ptr[rchild - inner_size];
      rchild = -(rchild - inner_size + 1);
    }
    else
    {
      r_aabb = inner_aabb_ptr[rchild];
      // do the offset now
      rchild *= 4;
    }
    vec1[0] = l_aabb.m_ranges[0].min ();
    vec1[1] = l_aabb.m_ranges[1].min ();
    vec1[2] = l_aabb.m_ranges[2].min ();

    vec1[3] = l_aabb.m_ranges[0].max ();
    vec2[0] = l_aabb.m_ranges[1].max ();
    vec2[1] = l_aabb.m_ranges[2].max ();

    vec2[2] = r_aabb.m_ranges[0].min ();
    vec2[3] = r_aabb.m_ranges[1].min ();
    vec3[0] = r_aabb.m_ranges[2].min ();

    vec3[1] = r_aabb.m_ranges[0].max ();
    vec3[2] = r_aabb.m_ranges[1].max ();
    vec3[3] = r_aabb.m_ranges[2].max ();

    const int32 out_offset = node * 4;
    flat_ptr[out_offset + 0] = vec1;
    flat_ptr[out_offset + 1] = vec2;
    flat_ptr[out_offset + 2] = vec3;

    constexpr int32 isize = sizeof (int32);
    // memcopy so we do not truncate the ints
    memcpy (&vec4[0], &lchild, isize);
    memcpy (&vec4[1], &rchild, isize);
    flat_ptr[out_offset + 3] = vec4;
  });
  DRAY_ERROR_CHECK();

  return flat_bvh;
}

BVH LinearBVHBuilder::construct (Array<AABB<>> aabbs)
{

  Array<int32> primitive_ids = array_counting (aabbs.size (), 0, 1);
  return construct (aabbs, primitive_ids);
}

BVH LinearBVHBuilder::construct (Array<AABB<>> aabbs, Array<int32> primitive_ids)
{
  DRAY_LOG_OPEN ("bvh_construct");
  DRAY_LOG_ENTRY ("num_aabbs", aabbs.size ());

  bool zero_case = aabbs.size() == 0;
  if(zero_case)
  {
    // Special case that we have to deal with due to
    // the internal bvh representation
    Array<AABB<>> new_aabbs;
    new_aabbs.resize (2);
    AABB<>  *new_ptr = nullptr;
    new_ptr = new_aabbs.get_host_ptr ();
    AABB<> invalid;
    Vec<float32,3> zero({0.f, 0.f, 0.f});
    invalid.include(zero);
    new_ptr[0] = invalid;
    new_ptr[1] = invalid;
    aabbs = new_aabbs;
    Array<int32> new_pids;
    new_pids.resize(2);
    int32 *new_pid_ptr = new_pids.get_host_ptr();
    new_pid_ptr[0] = 0;
    new_pid_ptr[1] = 0;
    primitive_ids = new_pids;
  }

  if (aabbs.size () == 1)
  {
    // Special case that we have to deal with due to
    // the internal bvh representation
    Array<AABB<>> new_aabbs;
    new_aabbs.resize (2);
    AABB<> *old_ptr = nullptr, *new_ptr = nullptr;
    old_ptr = aabbs.get_host_ptr ();
    new_ptr = new_aabbs.get_host_ptr ();
    new_ptr[0] = old_ptr[0];
    AABB<> invalid;
    Vec<float32,3> zero({0.f, 0.f, 0.f});
    invalid.include(zero);
    new_ptr[1] = invalid;
    aabbs = new_aabbs;
    Array<int32> new_pids;
    new_pids.resize(2);
    int32 *pid_ptr = primitive_ids.get_host_ptr();
    int32 *new_pid_ptr = new_pids.get_host_ptr();
    new_pid_ptr[0] = pid_ptr[0];
    new_pid_ptr[1] = pid_ptr[0];
    primitive_ids = new_pids;
  }
  Timer tot_time;
  Timer timer;

  AABB<> bounds;
  if(!zero_case)
  {
    bounds = reduce (aabbs);
  }
  DRAY_LOG_ENTRY ("reduce", timer.elapsed ());
  timer.reset ();

  Array<uint32> mcodes = get_mcodes (aabbs, bounds);
  DRAY_LOG_ENTRY ("morton_codes", timer.elapsed ());
  timer.reset ();

  // original positions of the sorted morton codes.
  // allows us to gather / sort other arrays.
  Array<int32> ids = sort_mcodes (mcodes);
  DRAY_LOG_ENTRY ("sort", timer.elapsed ());
  timer.reset ();

  reorder (ids, aabbs);
  reorder (ids, primitive_ids);
  DRAY_LOG_ENTRY ("reorder", timer.elapsed ());
  timer.reset ();

  const int size = aabbs.size ();

  BVHData bvh_data;
  // the arrays that already exist
  bvh_data.m_leafs = primitive_ids;
  bvh_data.m_mcodes = mcodes;
  bvh_data.m_leaf_aabbs = aabbs;
  // arrays we have to calculate
  bvh_data.m_inner_aabbs.resize (size - 1);
  bvh_data.m_left_children.resize (size - 1);
  bvh_data.m_right_children.resize (size - 1);
  bvh_data.m_parents.resize (size + size - 1);

  // assign parent and child pointers
  build_tree (bvh_data);
  DRAY_LOG_ENTRY ("build_tree", timer.elapsed ());
  timer.reset ();

  propagate_aabbs (bvh_data);
  DRAY_LOG_ENTRY ("propagate", timer.elapsed ());
  timer.reset ();


  BVH bvh;
  bvh.m_inner_nodes = emit (bvh_data);
  DRAY_LOG_ENTRY ("emit", timer.elapsed ());
  timer.reset ();

  bvh.m_leaf_nodes = bvh_data.m_leafs;
  bvh.m_bounds = bounds;
  bvh.m_aabb_ids = ids;

  DRAY_LOG_ENTRY ("tot_time", tot_time.elapsed ());
  DRAY_LOG_CLOSE ();
  return bvh;
}

} // namespace dray
