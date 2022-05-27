// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNSTRUCTURED_MESH_HPP
#define DRAY_UNSTRUCTURED_MESH_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/data_model/mesh.hpp>
#include <dray/data_model/element.hpp>
#include <dray/data_model/grid_function.hpp>
#include <dray/aabb.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/location.hpp>
#include <dray/newton_solver.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/vec.hpp>
#include <dray/error.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

template <int32 dim, ElemType etype, int32 P_order>
using MeshElem = Element<dim, 3u, etype, P_order>;
// forward declare so we can have template friend
template <typename ElemT> struct DeviceMesh;

/*
 * @class UnstructuredMesh
 * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
 *
 * @warning Triangle and tet meshes are broken until we change ref aabbs to SubRef<etype>
 *          and implement reference space splitting for reference simplices.
 */
template <class Element> class UnstructuredMesh : public Mesh
{
public:
  // Typedefs
  static constexpr auto dim = Element::get_dim ();
  static constexpr auto etype = Element::get_etype ();
  using ElementType = Element;

protected:
  GridFunction<3u> m_dof_data;
  int32 m_poly_order;
  bool m_is_constructed;
  // we are lazy constructing these
  BVH m_bvh;
  Array<SubRef<dim, etype>> m_ref_aabbs;

  //// Accept input data (as shared).
  //// Useful for keeping same data but changing class template arguments.
  //// If kept protected, can only be called by Mesh<Element> or friends of Mesh<Element>.
  //UnstructuredMesh(GridFunction<Element::get_ncomp()> dof_data,
  //                 int32 poly_order,
  //                 bool is_constructed,
  //                 BVH bvh,
  //                 Array<SubRef<dim, etype>> ref_aabbs)
  //  :
  //    m_dof_data(dof_data),
  //    m_poly_order(poly_order),
  //    m_is_constructed(is_constructed),
  //    m_bvh(bvh),
  //    m_ref_aabbs(ref_aabbs)

  //{ }

public:

  virtual ~UnstructuredMesh();

  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  virtual std::string type_name() const override;

  virtual AABB<3> bounds() override;
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) override;
  virtual void to_node(conduit::Node &n_topo) override;


  friend struct DeviceMesh<Element>;

  UnstructuredMesh () = delete;  // For now, probably need later.
  UnstructuredMesh (const GridFunction<3u> &dof_data, int32 poly_order);
  // ndofs=3u because mesh always lives in 3D, even if it is a surface.

  UnstructuredMesh(UnstructuredMesh &&other);
  UnstructuredMesh(const UnstructuredMesh &other);

  const BVH get_bvh ();

  GridFunction<3u> get_dof_data ()
  {
    return m_dof_data;
  }

  const GridFunction<3u> & get_dof_data() const
  {
    return m_dof_data;
  }

  const Array<SubRef<dim, etype>> &get_ref_aabbs () const
  {
    return m_ref_aabbs;
  }

}; // Mesh

// Element<topo dims, ncomps, base_shape, polynomial order>
using Hex3   = Element<3u, 3u, ElemType::Tensor, Order::General>;
using Hex_P1 = Element<3u, 3u, ElemType::Tensor, Order::Linear>;
using Hex_P2 = Element<3u, 3u, ElemType::Tensor, Order::Quadratic>;

using Tet3   = Element<3u, 3u, ElemType::Simplex, Order::General>;
using Tet_P1 = Element<3u, 3u, ElemType::Simplex, Order::Linear>;
using Tet_P2 = Element<3u, 3u, ElemType::Simplex, Order::Quadratic>;

using Quad3   = Element<2u, 3u,ElemType::Tensor, Order::General>;
using Quad_P1 = Element<2u, 3u,ElemType::Tensor, Order::Linear>;
using Quad_P2 = Element<2u, 3u,ElemType::Tensor, Order::Quadratic>;

using Tri3    = Element<2u, 3u, ElemType::Simplex, Order::General>;
using Tri_P1  = Element<2u, 3u, ElemType::Simplex, Order::Linear>;
using Tri_P2  = Element<2u, 3u, ElemType::Simplex, Order::Quadratic>;

using HexMesh = UnstructuredMesh<Hex3>;
using HexMesh_P1 = UnstructuredMesh<Hex_P1>;
using HexMesh_P2 = UnstructuredMesh<Hex_P2>;

using TetMesh = UnstructuredMesh<Tet3>;
using TetMesh_P1 = UnstructuredMesh<Tet_P1>;
using TetMesh_P2 = UnstructuredMesh<Tet_P2>;

using QuadMesh = UnstructuredMesh<Quad3>;
using QuadMesh_P1 = UnstructuredMesh<Quad_P1>;
using QuadMesh_P2 = UnstructuredMesh<Quad_P2>;

using TriMesh = UnstructuredMesh<Tri3>;
using TriMesh_P1 = UnstructuredMesh<Tri_P1>;
using TriMesh_P2 = UnstructuredMesh<Tri_P2>;

} // namespace dray

#endif // DRAY_MESH_HPP
