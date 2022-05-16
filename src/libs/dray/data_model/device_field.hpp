// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DEVICE_FIELD_HPP
#define DRAY_DEVICE_FIELD_HPP

#include <dray/data_model/element.hpp>
#include <dray/data_model/grid_function.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/exports.hpp>
#include <dray/vec.hpp>

namespace dray
{
/*
 * @class FieldAccess
 * @brief Device-safe access to a collection of elements (just knows about the geometry, not fields).
 */
template <class ElemT> struct DeviceField
{
  static constexpr auto dim = ElemT::get_dim ();
  static constexpr auto ncomp = ElemT::get_ncomp ();
  static constexpr auto etype = ElemT::get_etype ();

  const int32 *m_idx_ptr;
  const Vec<Float, ncomp> *m_val_ptr;
  const int32 m_poly_order;

  //TODO use a DeviceGridFunction

  DeviceField() = delete;
  DeviceField(UnstructuredField<ElemT> &field);

  DRAY_EXEC typename AdaptGetOrderPolicy<ElemT>::type get_order_policy() const
  {
    return adapt_get_order_policy(ElemT{}, m_poly_order);
  }

  DRAY_EXEC ElemT get_elem (int32 el_idx) const;
};

template<class ElemT>
DeviceField<ElemT>::DeviceField(UnstructuredField<ElemT> &field)
  : m_idx_ptr(field.m_dof_data.m_ctrl_idx.get_device_ptr_const()),
    m_val_ptr(field.m_dof_data.m_values.get_device_ptr_const()),
    m_poly_order(field.m_poly_order)
{
}

template <class ElemT>
DRAY_EXEC ElemT DeviceField<ElemT>::get_elem (int32 el_idx) const
{
  // We are just going to assume that the elements in the data store
  // are in the same position as their id, el_id==el_idx.
  ElemT ret;

  auto shape = adapt_get_shape(ElemT{});
  auto order_p = get_order_policy();
  const int32 dofs_per  = eattr::get_num_dofs(shape, order_p);

  SharedDofPtr<Vec<Float, ncomp>> dof_ptr{ dofs_per * el_idx + m_idx_ptr,
                                           m_val_ptr };
  ret.construct (el_idx, dof_ptr, m_poly_order);
  return ret;
}

} // namespace dray
#endif // DRAY_FIELD_HPP
