// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/data_model/element.hpp>
#include <dray/data_model/subref.hpp>
#include <dray/data_model/elem_ops.hpp>  // tri/tet dof indexing
#include <dray/data_model/detached_element.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/aabb.hpp>
#include <dray/array_utils.hpp>
#include <dray/dispatcher.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>


namespace dray
{

namespace detail
{

DRAY_EXEC void swap (int32 &a, int32 &b)
{
  int32 tmp;
  tmp = a;
  a = b;
  b = tmp;
}

DRAY_EXEC void sort4 (Vec<int32, 4> &vec)
{
  if (vec[0] > vec[1])
  {
    swap (vec[0], vec[1]);
  }
  if (vec[2] > vec[3])
  {
    swap (vec[2], vec[3]);
  }
  if (vec[0] > vec[2])
  {
    swap (vec[0], vec[2]);
  }
  if (vec[1] > vec[3])
  {
    swap (vec[1], vec[3]);
  }
  if (vec[1] > vec[2])
  {
    swap (vec[1], vec[2]);
  }
}

DRAY_EXEC void sort3 (Vec<int32, 4> &vec)
{
  if (vec[0] > vec[1])
  {
    swap (vec[0], vec[1]);
  }
  if (vec[0] > vec[2])
  {
    swap (vec[0], vec[2]);
  }
  if (vec[1] > vec[2])
  {
    swap (vec[1], vec[2]);
  }
}

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

Array<int32> sort_faces (Array<Vec<int32, 4>> &faces)
{
  const int size = faces.size ();
  Array<int32> iter = array_counting (size, 0, 1);
  // TODO: create custom sort for GPU / CPU
  int32 *iter_ptr = iter.get_host_ptr ();
  Vec<int32, 4> *faces_ptr = faces.get_host_ptr ();

  std::sort (iter_ptr, iter_ptr + size, [=] (int32 i1, int32 i2) {
    const Vec<int32, 4> f1 = faces_ptr[i1];
    const Vec<int32, 4> f2 = faces_ptr[i2];
    if (f1[0] == f2[0])
    {
      if (f1[1] == f2[1])
      {
        if (f1[2] == f2[2])
        {
          return f1[3] < f2[3];
        }
        else
        {
          return f1[2] < f2[2];
        }
      }
      else
      {
        return f1[1] < f2[1];
      }
    }
    else
    {
      return f1[0] < f2[0];
    }

    // return true;
  });


  reorder (iter, faces);
  return iter;
}

DRAY_EXEC bool is_same (const Vec<int32, 4> &a, const Vec<int32, 4> &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}

void unique_faces (Array<Vec<int32, 4>> &faces, Array<int32> &orig_ids)
{
  const int32 size = faces.size ();

  Array<int32> unique_flags;
  unique_flags.resize (size);
  // assum everyone is unique
  array_memset (unique_flags, 1);

  const Vec<int32, 4> *faces_ptr = faces.get_device_ptr_const ();
  int32 *unique_flags_ptr = unique_flags.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    // we assume everthing is sorted and there can be at most
    // two faces that can be shared
    const Vec<int32, 4> me = faces_ptr[i];
    bool duplicate = false;
    if (i != 0)
    {
      const Vec<int32, 4> left = faces_ptr[i - 1];
      if (is_same (me, left))
      {
        duplicate = true;
      }
    }
    if (i != size - 1)
    {
      const Vec<int32, 4> right = faces_ptr[i + 1];
      if (is_same (me, right))
      {
        duplicate = true;
      }
    }

    if (duplicate)
    {
      // mark myself for death
      unique_flags_ptr[i] = 0;
    }
  });
  DRAY_ERROR_CHECK();
  faces = index_flags (unique_flags, faces);
  orig_ids = index_flags (unique_flags, orig_ids);
}

// extract_faces (Hex -> Tensor)
template <int32 ncomp, int32 P>
Array<Vec<int32, 4>> extract_faces(UnstructuredMesh<Element<3, ncomp, ElemType::Tensor, P>> &mesh)
{
  using ElemT = Element<3, ncomp, ElemType::Tensor, P>;

  const int num_els = mesh.cells ();

  Array<Vec<int32, 4>> faces;
  faces.resize (num_els * 6);
  Vec<int32, 4> *faces_ptr = faces.get_device_ptr ();

  DeviceMesh<ElemT> device_mesh (mesh);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_els), [=] DRAY_LAMBDA (int32 el_id) {
    // assume that if one dof is shared on a face then all dofs are shares.
    // if this is not the case this is a much harder problem
    const int32 p = device_mesh.m_poly_order;
    const int32 stride_y = p + 1;
    const int32 stride_z = stride_y * stride_y;
    const int32 el_offset = stride_z * stride_y * el_id;
    const int32 *el_ptr = device_mesh.m_idx_ptr + el_offset;
    int32 corners[8];
    corners[0] = el_ptr[0];
    corners[1] = el_ptr[p];
    corners[2] = el_ptr[stride_y * p];
    corners[3] = el_ptr[stride_y * p + p];
    corners[4] = el_ptr[stride_z * p];
    corners[5] = el_ptr[stride_z * p + p];
    corners[6] = el_ptr[stride_z * p + stride_y * p];
    corners[7] = el_ptr[stride_z * p + stride_y * p + p];

    // I think this is following masado's conventions
    Vec<int32, 4> face;

    // x
    face[0] = corners[0];
    face[1] = corners[2];
    face[2] = corners[4];
    face[3] = corners[6];
    sort4 (face);

    faces_ptr[el_id * 6 + 0] = face;
    // X
    face[0] = corners[1];
    face[1] = corners[3];
    face[2] = corners[5];
    face[3] = corners[7];

    sort4 (face);
    faces_ptr[el_id * 6 + 3] = face;

    // y
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[4];
    face[3] = corners[5];

    sort4 (face);
    faces_ptr[el_id * 6 + 1] = face;
    // Y
    face[0] = corners[2];
    face[1] = corners[3];
    face[2] = corners[6];
    face[3] = corners[7];

    sort4 (face);
    faces_ptr[el_id * 6 + 4] = face;

    // z
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[2];
    face[3] = corners[3];

    sort4 (face);
    faces_ptr[el_id * 6 + 2] = face;

    // Z
    face[0] = corners[4];
    face[1] = corners[5];
    face[2] = corners[6];
    face[3] = corners[7];

    sort4 (face);
    faces_ptr[el_id * 6 + 5] = face;
  });
  DRAY_ERROR_CHECK();

  return faces;
}

// extract_faces (Tet -> Simplex)
template <int32 ncomp, int32 P>
Array<Vec<int32, 4>> extract_faces(UnstructuredMesh<Element<3, ncomp, ElemType::Simplex, P>> &mesh)
{
  using ElemT = Element<3, ncomp, ElemType::Simplex, P>;

  const int num_els = mesh.cells ();

  Array<Vec<int32, 4>> faces;
  faces.resize (num_els * 4);
  Vec<int32, 4> *faces_ptr = faces.get_device_ptr ();

  DeviceMesh<ElemT> device_mesh (mesh);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_els), [=] DRAY_LAMBDA (int32 el_id) {
    // assume that if one dof is shared on a face then all dofs are shares.
    // if this is not the case this is a much harder problem
    auto order_p = device_mesh.get_order_policy();
    const int32 p = eattr::get_order(order_p);
    const int32 el_offset = el_id * eattr::get_num_dofs(ShapeTet{}, order_p);
    const int32 *el_ptr = device_mesh.m_idx_ptr + el_offset;

    int32 corners[4];
    corners[0] = el_ptr[detail::cartesian_to_tet_idx(p, 0, 0, p+1)];
    corners[1] = el_ptr[detail::cartesian_to_tet_idx(0, p, 0, p+1)];
    corners[2] = el_ptr[detail::cartesian_to_tet_idx(0, 0, p, p+1)];
    corners[3] = el_ptr[0];

    // The reference tetrahedron. Vertex v3 is at the origin.
    //
    //  Front:
    //          (z)
    //          v2
    //         /. \_
    //        / .  \_
    //       /.v3.  \_
    //     v0______`_v1
    //   (x)          (y)
    //
    //
    //   =========     =========      =========      =========
    //   face id 0     face id 1      face id 2      face id 3
    //
    //   (2)           (2)            (1)            (1)
    //    z             z              y              y
    //    |\            |\             |\             |\
    //    | \           | \            | \            | \
    //    o__y          o__x           o__x           z__x
    //  (3)  (1)      (3)  (0)       (3)  (0)       (2)  (0)
    //

    Vec<int32, 4> face;
    face[3] = -1;            // Only use the first 3 indices

    // yzo (opposite x)
    face[0] = corners[1];
    face[1] = corners[2];
    face[2] = corners[3];
    sort3 (face);
    faces_ptr[el_id * 4 + 0] = face;

    // xzo (opposite y)
    face[0] = corners[0];
    face[1] = corners[2];
    face[2] = corners[3];
    sort3 (face);
    faces_ptr[el_id * 4 + 1] = face;

    // xyo (opposite z)
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[3];
    sort3 (face);
    faces_ptr[el_id * 4 + 2] = face;

    // xyz (opposite o)
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[2];
    sort3 (face);
    faces_ptr[el_id * 4 + 3] = face;
  });
  DRAY_ERROR_CHECK();

  return faces;
}

// Returns faces, where faces[i][0] = el_id and 0 <= faces[i][1] = face_id < 6.
template <ElemType etype>
Array<Vec<int32, 2>> reconstruct (Array<int32> &orig_ids)
{
  const int32 size = orig_ids.size ();

  Array<Vec<int32, 2>> face_ids;
  face_ids.resize (size);

  const int32 *orig_ids_ptr = orig_ids.get_device_ptr_const ();
  Vec<int32, 2> *faces_ids_ptr = face_ids.get_device_ptr ();

  static_assert((etype == ElemType::Tensor || etype == ElemType::Simplex),
      "Unknown element type");

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    constexpr int32 num_faces_per_elem = (etype == ElemType::Tensor ? 6
                                          : etype == ElemType::Simplex ? 4
                                          : -1);
    const int32 flat_id = orig_ids_ptr[i];
    const int32 el_id = flat_id / num_faces_per_elem;
    const int32 face_id = flat_id % num_faces_per_elem;
    Vec<int32, 2> face;
    face[0] = el_id;
    face[1] = face_id;
    faces_ids_ptr[i] = face;
  });
  DRAY_ERROR_CHECK();
  return face_ids;
}

// TODO
/// template<typename T, class ElemT>
/// BVH construct_face_bvh(Mesh<T, ElemT> &mesh, Array<Vec<int32,2>> &faces)
/// {
///   constexpr double bbox_scale = 1.000001;
///   const int32 num_faces = faces.size();
///   Array<AABB<>> aabbs;
///   aabbs.resize(num_faces);
///   AABB<> *aabb_ptr = aabbs.get_device_ptr();
///
///   MeshAccess<T, ElemT> device_mesh = mesh.access_device_mesh();
///   const Vec<int32,2> *faces_ptr = faces.get_device_ptr_const();
///
///   RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_faces), [=] DRAY_LAMBDA (int32 face_id)
///   {
///     const Vec<int32,2> face = faces_ptr[face_id];
///     FaceElement<T,3> face_elem = device_mesh.get_elem(face[0]).get_face_element(face[1]);
///
///     AABB<> bounds;
///     face_elem.get_bounds(bounds);
///     bounds.scale(bbox_scale);
///     aabb_ptr[face_id] = bounds;
///   });
///
///   LinearBVHBuilder builder;
///   BVH bvh = builder.construct(aabbs);
///   return bvh;
/// }

// TODO
/// template<typename T, class ElemT>
/// typename Mesh<T, ElemT>::ExternalFaces  external_faces(Mesh<T, ElemT> &mesh)
/// {
///   Array<Vec<int32,4>> faces = extract_faces(mesh);
///
///   Array<int32> orig_ids = sort_faces(faces);
///   unique_faces(faces, orig_ids);
///
///
///   const int num_els = mesh.cells();
///   Array<Vec<int32,2>> res = reconstruct(orig_ids);
///
///   BVH bvh = construct_face_bvh(mesh, res);
///
///   typename Mesh<T, ElemT>::ExternalFaces ext_faces;
///   ext_faces.m_faces = res;
///   ext_faces.m_bvh = bvh;
///   return ext_faces;
/// }


namespace detail
{
  template <int32 dim>
  DRAY_EXEC
  Split<ElemType::Simplex> pick_binary_split(const RefSpaceTag<dim, ElemType::Simplex>,
                                         const SubRef<dim, ElemType::Simplex> &subtri)
  {
    // Pick the longest edge, out of all edges v1 < v2.
    int32 max_v1 = 0, max_v2 = 1;
    Float max_length2 = 0;
    for (int32 v1 = 0; v1 < dim; ++v1)
      for (int32 v2 = v1+1; v2 < dim+1; ++v2)
      {
        const Float length2 = (subtri[v1] - subtri[v2]).magnitude2();
        if (length2 > max_length2)
        {
          max_v1 = v1;
          max_v2 = v2;
          max_length2 = length2;
        }
      }

    // Split in the center of the edge.
    Split<ElemType::Simplex> split;
    split.vtx_displaced = max_v1;    // Arbitrary choice (other side is complement).
    split.vtx_tradeoff = max_v2;
    split.factor = 0.5;

    return split;
  }

  template <int32 dim>
  DRAY_EXEC
  Split<ElemType::Tensor> pick_binary_split(const RefSpaceTag<dim, ElemType::Tensor>,
                                          const SubRef<dim, ElemType::Tensor> &subcube)
  {
    int32 max_dim = 0;
    Float max_length = subcube[1][0] - subcube[0][0];
    for (int32 d = 1; d < dim; ++d)
    {
      const Float length = subcube[1][d] - subcube[0][d];
      if (length > max_length)
      {
        max_dim = d;
        max_length = length;
      }
    }

    // Split in the center of the axis.
    Split<ElemType::Tensor> split;
    split.axis = max_dim;
    split.f_lower_t_upper = false;  // Arbitrary choice (other side is complement).
    split.factor = 0.5;

    return split;
  }
}






template <class ElemT>
BVH construct_bvh (UnstructuredMesh<ElemT> &mesh, Array<typename get_subref<ElemT>::type> &ref_aabbs)
{
  DRAY_LOG_OPEN ("construct_bvh");

  constexpr double bbox_scale = 1.000001;
  // this has to be inside the lambda for gcc8.1 otherwise:
  // error: use of 'this' in a constant expression
  // so we add another definition
  constexpr uint32 dim_outside = ElemT::get_dim ();
  constexpr auto etype_outside = ElemT::get_etype ();

  const int num_els = mesh.cells();

  constexpr int splits = 2 * (2 << dim_outside);
  const int32 num_scratch_els = num_els * (splits + 1);
  // TODO: Note splits are no longer controllable
  using ShapeTag = typename AdaptGetShape<ElemT>::type;
  using OrderPolicy = typename AdaptGetOrderPolicy<ElemT>::type;
  const OrderPolicy order_p = adapt_get_order_policy(ElemT(), mesh.order());
  const size_t nodes_per_elem = eattr::get_num_dofs(ShapeTag(), order_p);

  Array<AABB<>> aabbs;
  Array<int32> prim_ids;

  aabbs.resize (num_els * (splits + 1));
  prim_ids.resize (num_els * (splits + 1));
  ref_aabbs.resize (num_els * (splits + 1));

  AABB<> *aabb_ptr = aabbs.get_device_ptr ();
  int32 *prim_ids_ptr = prim_ids.get_device_ptr ();
  SubRef<dim_outside, etype_outside> *ref_aabbs_ptr = ref_aabbs.get_device_ptr ();

  // ask for a device mesh without the bvh, which we are building
  DeviceMesh<ElemT> device_mesh (mesh, false);

  GridFunction<3> split_scratch_gf;
  split_scratch_gf.resize_counting(num_scratch_els, nodes_per_elem);
  const int32 * split_scratch_idx_ptr = split_scratch_gf.m_ctrl_idx.get_device_ptr_const();
  Vec<Float, 3> * split_scratch_val_ptr = split_scratch_gf.m_values.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_els), [=] DRAY_LAMBDA (int32 el_id) {

    constexpr uint32 dim = ElemT::get_dim ();
    constexpr uint32 ncomp = ElemT::get_ncomp();
    constexpr auto etype = ElemT::get_etype ();
    const RefSpaceTag<dim, etype> ref_space_tag;

    const ElemT this_elem_tag = device_mesh.get_elem(el_id);
    const int32 p_order = this_elem_tag.get_order();

    AABB<> boxs[splits + 1];
    SubRef<dim, etype> ref_boxs[splits + 1];
    const int32 * el_split_scratch_idx = split_scratch_idx_ptr + el_id * (splits+1) * nodes_per_elem;
    AABB<> tot;

    device_mesh.get_elem (el_id).get_bounds (boxs[0]);
    tot = boxs[0];
    ref_boxs[0] = ref_universe(ref_space_tag);
    int32 count = 1;

    // Populate position 0 scratch space coords with original coords.
    {
      WriteDofPtr<Vec<Float, ncomp>> wdp_original;
      wdp_original.m_offset_ptr = el_split_scratch_idx;
      wdp_original.m_dof_ptr = split_scratch_val_ptr;
      for (int32 nidx = 0; nidx < nodes_per_elem; ++nidx)
        wdp_original[nidx] = this_elem_tag.read_dof_ptr()[nidx];
    }

    for (int i = 0; i < splits; ++i)
    {
      // find split
      int32 max_id = 0;
      float32 max_measure = boxs[0].volume();
      if(max_measure == 0.f)
      {
        max_measure = boxs[0].surface_area();
      }
      for (int b = 1; b < count; ++b)
      {
        float32 measure = boxs[b].volume();
        if(measure == 0.f)
        {
          measure = boxs[b].surface_area();
        }
        if (measure > max_measure)
        {
          max_id = b;
          max_measure = measure;
        }
      }

      // Get a splitter by which to split ref and coeffs.
      Split<etype> splitter = detail::pick_binary_split(ref_space_tag, ref_boxs[max_id]);

      // Split ref box using splitter.
      //   In-place: Same side that is in-place for coeffs.
      //   Returns: The complement, so use the complement splitter for coeffs.
      ref_boxs[count] = split_subref(ref_boxs[max_id], splitter);

      // Split coefficients using splitter.
      //   First copy, then split each side corresponding to subref.
      WriteDofPtr<Vec<Float, ncomp>> wdp_mother, wdp_dghter;
      {
        wdp_mother.m_offset_ptr = el_split_scratch_idx + max_id * nodes_per_elem;
        wdp_dghter.m_offset_ptr = el_split_scratch_idx + count * nodes_per_elem;
        wdp_mother.m_dof_ptr = split_scratch_val_ptr;
        wdp_dghter.m_dof_ptr = split_scratch_val_ptr;
        for (int32 nidx = 0; nidx < nodes_per_elem; ++nidx)
          wdp_dghter[nidx] = wdp_mother[nidx];
      }
      split_inplace(this_elem_tag, wdp_mother, splitter);
      split_inplace(this_elem_tag, wdp_dghter, splitter.get_complement());

      // udpate the phys bounds
      { ElemT free_elem = ElemT::create(-1, wdp_mother.to_readonly_dof_ptr(), p_order);
        free_elem.get_bounds(boxs[max_id]);
      }
      { ElemT free_elem = ElemT::create(-1, wdp_dghter.to_readonly_dof_ptr(), p_order);
        free_elem.get_bounds(boxs[count]);
      }
      count++;
    }

    AABB<> res;
    for (int i = 0; i < splits + 1; ++i)
    {
      boxs[i].scale (bbox_scale);
      res.include (boxs[i]);
      aabb_ptr[el_id * (splits + 1) + i] = boxs[i];
      prim_ids_ptr[el_id * (splits + 1) + i] = el_id;
      ref_aabbs_ptr[el_id * (splits + 1) + i] = ref_boxs[i];
    }

    // if(el_id > 100 && el_id < 200)
    //{
    //  printf("cell id %d AREA %f %f diff %f\n",
    //                                 el_id,
    //                                 tot.volume(),
    //                                 res.volume(),
    //                                 tot.volume() - res.volume());
    //  //AABB<> ol =  tot.intersect(res);
    //  //float32 overlap =  ol.volume();

    //  //printf("overlap %f\n", overlap);
    //  //printf("%f %f %f - %f %f %f\n",
    //  //      tot.m_ranges[0].min(),
    //  //      tot.m_ranges[1].min(),
    //  //      tot.m_ranges[2].min(),
    //  //      tot.m_ranges[0].max(),
    //  //      tot.m_ranges[1].max(),
    //  //      tot.m_ranges[2].max());
    //}
  });
  DRAY_ERROR_CHECK();

  LinearBVHBuilder builder;
  BVH bvh = builder.construct (aabbs, prim_ids);
  DRAY_LOG_CLOSE ();
  return bvh;
}

} // namespace detail

} // namespace dray


//
// Explicit instantiations.
//
namespace dray
{
namespace detail
{
//
// reorder();
//
template void reorder (Array<int32> &indices, Array<float32> &array);
template void reorder (Array<int32> &indices, Array<float64> &array);


template Array<Vec<int32, 2>> reconstruct<ElemType::Simplex>(Array<int32> &orig_ids);
template Array<Vec<int32, 2>> reconstruct<ElemType::Tensor>(Array<int32> &orig_ids);


//
// extract_faces();
//
template Array<Vec<int32, 4>>
extract_faces(UnstructuredMesh<Element<3, 3, ElemType::Tensor, Order::General>> &mesh);
template Array<Vec<int32, 4>>
extract_faces(UnstructuredMesh<Element<3, 3, ElemType::Tensor, Order::Linear>> &mesh);
template Array<Vec<int32, 4>>
extract_faces(UnstructuredMesh<Element<3, 3, ElemType::Tensor, Order::Quadratic>> &mesh);

template Array<Vec<int32, 4>>
extract_faces(UnstructuredMesh<Element<3, 3, ElemType::Simplex, Order::General>> &mesh);
template Array<Vec<int32, 4>>
extract_faces(UnstructuredMesh<Element<3, 3, ElemType::Simplex, Order::Linear>> &mesh);
template Array<Vec<int32, 4>>
extract_faces(UnstructuredMesh<Element<3, 3, ElemType::Simplex, Order::Quadratic>> &mesh);


//
// construct_bvh();   // Tensor
//
template BVH construct_bvh (UnstructuredMesh<MeshElem<2, ElemType::Tensor, Order::General>> &mesh,
                            Array<SubRef<2, ElemType::Tensor>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<2, ElemType::Tensor, Order::Linear>> &mesh,
                            Array<SubRef<2, ElemType::Tensor>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<2, ElemType::Tensor, Order::Quadratic>> &mesh,
                            Array<SubRef<2, ElemType::Tensor>> &ref_aabbs);

template BVH construct_bvh (UnstructuredMesh<MeshElem<3, ElemType::Tensor, Order::General>> &mesh,
                            Array<SubRef<3, ElemType::Tensor>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<3, ElemType::Tensor, Order::Linear>> &mesh,
                            Array<SubRef<3, ElemType::Tensor>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<3, ElemType::Tensor, Order::Quadratic>> &mesh,
                            Array<SubRef<3, ElemType::Tensor>> &ref_aabbs);

//
// construct_bvh();   // Simplex
//
template BVH construct_bvh (UnstructuredMesh<MeshElem<2, ElemType::Simplex, Order::General>> &mesh,
                            Array<SubRef<2, ElemType::Simplex>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<2, ElemType::Simplex, Order::Linear>> &mesh,
                            Array<SubRef<2, ElemType::Simplex>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<2, ElemType::Simplex, Order::Quadratic>> &mesh,
                            Array<SubRef<2, ElemType::Simplex>> &ref_aabbs);

template BVH construct_bvh (UnstructuredMesh<MeshElem<3, ElemType::Simplex, Order::General>> &mesh,
                            Array<SubRef<3, ElemType::Simplex>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<3, ElemType::Simplex, Order::Linear>> &mesh,
                            Array<SubRef<3, ElemType::Simplex>> &ref_aabbs);
template BVH construct_bvh (UnstructuredMesh<MeshElem<3, ElemType::Simplex, Order::Quadratic>> &mesh,
                            Array<SubRef<3, ElemType::Simplex>> &ref_aabbs);

struct GetDofDataFunctor
{
  GetDofDataFunctor() = default;
  ~GetDofDataFunctor() = default;

  GridFunction<3> output() { return m_output; }

  template<typename MeshType>
  void operator()(MeshType &mesh)
  {
    GridFunction<3> temp = mesh.get_dof_data();
    m_output = temp;
  }

  GridFunction<3> m_output;
};

GridFunction<3>
get_dof_data(Mesh *mesh)
{
  GetDofDataFunctor func;
  dispatch(mesh, func);

  return func.output();
}

} // namespace detail
} // namespace dray
