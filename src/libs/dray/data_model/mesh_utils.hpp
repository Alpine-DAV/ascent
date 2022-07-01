// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MESH_UTILS_HPP
#define DRAY_MESH_UTILS_HPP

#include <dray/data_model/mesh.hpp>
#include <dray/data_model/subref.hpp>
#include <dray/data_model/elem_ops.hpp>

namespace dray
{

namespace detail
{

DRAY_EXEC void swap (int32 &a, int32 &b);

DRAY_EXEC void sort4 (Vec<int32, 4> &vec);

template <typename T> void reorder (Array<int32> &indices, Array<T> &array);

Array<int32> sort_faces (Array<Vec<int32, 4>> &faces);

DRAY_EXEC bool is_same (const Vec<int32, 4> &a, const Vec<int32, 4> &b);

void unique_faces (Array<Vec<int32, 4>> &faces, Array<int32> &orig_ids);

// Returns 6 (4) faces for each hex (tet) element, each face
// represented by the ids of the corner dofs.
template <int32 ncomp, int32 P>
Array<Vec<int32, 4>> extract_faces(UnstructuredMesh<Element<3, ncomp, ElemType::Tensor, P>> &mesh);

template <int32 ncomp, int32 P>
Array<Vec<int32, 4>> extract_faces(UnstructuredMesh<Element<3, ncomp, ElemType::Simplex, P>> &mesh);



// Returns faces, where faces[i][0] = el_id and 0 <= faces[i][1] = face_id < 6.
// This allows us to identify the needed dofs for a face mesh.
template <ElemType etype>
Array<Vec<int32, 2>> reconstruct (Array<int32> &orig_ids);

// TODO
/// template<typename T, class ElemT>
/// BVH construct_face_bvh(Mesh<T, ElemT> &mesh, Array<Vec<int32,2>> &faces);

// TODO
/// template<typename T, class ElemT>
/// typename Mesh<T, ElemT>::ExternalFaces  external_faces(Mesh<T, ElemT> &mesh);

template <class ElemT>
BVH construct_bvh (UnstructuredMesh<ElemT> &mesh, Array<typename get_subref<ElemT>::type> &ref_aabbs);

} // namespace detail

} // namespace dray


#endif // DRAY_MESH_UTILS_HPP
