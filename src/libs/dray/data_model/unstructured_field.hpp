// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNSTRUCTURED_FIELD_HPP
#define DRAY_UNSTRUCTURED_FIELD_HPP

#include <dray/data_model/element.hpp>
#include <dray/data_model/grid_function.hpp>
#include <dray/data_model/field.hpp>
#include <dray/exports.hpp>
#include <dray/vec.hpp>
#include <dray/error.hpp>

namespace dray
{

template <int32 dim, int32 ncomp, ElemType etype, int32 P_Order>
using FieldElem = Element<dim, ncomp, etype, P_Order>;

// forward declare so we can have template friend
template <typename ElemT> struct DeviceField;


/*
 * @class Field
 * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
 */
template <class ElemT> class UnstructuredField : public Field
{
  protected:
  GridFunction<ElemT::get_ncomp ()> m_dof_data;
  int32 m_poly_order;
  mutable bool m_range_calculated;
  mutable std::vector<Range> m_ranges;

  public:
  UnstructuredField () = delete; // For now, probably need later.
  UnstructuredField (const GridFunction<ElemT::get_ncomp ()>
         &dof_data, int32 poly_order,
         const std::string name = "");

  friend struct DeviceField<ElemT>;

  UnstructuredField(UnstructuredField &&other);
  UnstructuredField(const UnstructuredField &other);

  virtual void to_node(conduit::Node &n_field) override;

  virtual int32 order() const override;

  virtual void eval(const Array<Location> locs, Array<Float> &values) override;

  int32 get_poly_order () const
  {
    return m_poly_order;
  }

  virtual int32 components() const override
  {
    return ElemT::get_ncomp();
  }

  int32 get_num_elem () const
  {
    return m_dof_data.get_num_elem ();
  }

  GridFunction<ElemT::get_ncomp ()> get_dof_data ()
  {
    return m_dof_data;
  }

  const GridFunction<ElemT::get_ncomp ()> & get_dof_data () const
  {
    return m_dof_data;
  }

  virtual std::vector<Range> range () const override;

  virtual std::string type_name() const override;

  static UnstructuredField uniform_field(int32 num_els,
                             const Vec<Float, ElemT::get_ncomp()> &val,
                             const std::string &name = "");
};

// Element<topo dims, ncomps, base_shape, polynomial order>
using HexScalar  = Element<3u, 1u, ElemType::Tensor, Order::General>;
using HexScalar_P0  = Element<3u, 1u, ElemType::Tensor, Order::Constant>;
using HexScalar_P1  = Element<3u, 1u, ElemType::Tensor, Order::Linear>;
using HexScalar_P2  = Element<3u, 1u, ElemType::Tensor, Order::Quadratic>;

using TetScalar  = Element<3u, 1u, ElemType::Simplex, Order::General>;
using TetScalar_P0 = Element<3u, 1u, ElemType::Simplex, Order::Constant>;
using TetScalar_P1 = Element<3u, 1u, ElemType::Simplex, Order::Linear>;
using TetScalar_P2 = Element<3u, 1u, ElemType::Simplex, Order::Quadratic>;

using QuadScalar  = Element<2u, 1u, ElemType::Tensor, Order::General>;
using QuadScalar_P0 = Element<2u, 1u, ElemType::Tensor, Order::Constant>;
using QuadScalar_P1 = Element<2u, 1u, ElemType::Tensor, Order::Linear>;
using QuadScalar_P2 = Element<2u, 1u, ElemType::Tensor, Order::Quadratic>;

using TriScalar  = Element<2u, 1u, ElemType::Simplex, Order::General>;
using TriScalar_P0 = Element<2u, 1u, ElemType::Simplex, Order::Constant>;
using TriScalar_P1 = Element<2u, 1u, ElemType::Simplex, Order::Linear>;
using TriScalar_P2 = Element<2u, 1u, ElemType::Simplex, Order::Quadratic>;


using HexVector = Element<3u, 3u, ElemType::Tensor, Order::General>;
using HexVector_P0 = Element<3u, 3u, ElemType::Tensor, Order::Constant>;
using HexVector_P1 = Element<3u, 3u, ElemType::Tensor, Order::Linear>;
using HexVector_P2 = Element<3u, 3u, ElemType::Tensor, Order::Quadratic>;

using QuadVector = Element<2u, 3u,ElemType::Tensor, Order::General>;
using QuadVector_P0 = Element<2u, 3u,ElemType::Tensor, Order::Constant>;
using QuadVector_P1 = Element<2u, 3u,ElemType::Tensor, Order::Linear>;
using QuadVector_P2 = Element<2u, 3u,ElemType::Tensor, Order::Quadratic>;

using QuadVector_2D = Element<2u, 2u,ElemType::Tensor, Order::General>;
using QuadVector_2D_P0 = Element<2u, 2u,ElemType::Tensor, Order::Constant>;
using QuadVector_2D_P1 = Element<2u, 2u,ElemType::Tensor, Order::Linear>;
using QuadVector_2D_P2 = Element<2u, 2u,ElemType::Tensor, Order::Quadratic>;

using TetVector = Element<3u, 3u, ElemType::Simplex, Order::General>;
using TetVector_P0 = Element<3u, 3u, ElemType::Simplex, Order::Constant>;
using TetVector_P1 = Element<3u, 3u, ElemType::Simplex, Order::Linear>;
using TetVector_P2 = Element<3u, 3u, ElemType::Simplex, Order::Quadratic>;

using TriVector = Element<2u, 3u,ElemType::Simplex, Order::General>;
using TriVector_P0 = Element<2u, 3u,ElemType::Simplex, Order::Constant>;
using TriVector_P1 = Element<2u, 3u,ElemType::Simplex, Order::Linear>;
using TriVector_P2 = Element<2u, 3u,ElemType::Simplex, Order::Quadratic>;

using TriVector_2D = Element<2u, 2u,ElemType::Simplex, Order::General>;
using TriVector_2D_P0 = Element<2u, 2u,ElemType::Simplex, Order::Constant>;
using TriVector_2D_P1 = Element<2u, 2u,ElemType::Simplex, Order::Linear>;
using TriVector_2D_P2 = Element<2u, 2u,ElemType::Simplex, Order::Quadratic>;



} // namespace dray
#endif // DRAY_FIELD_HPP
