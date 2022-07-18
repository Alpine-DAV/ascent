// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_GRID_FUNCTION_DATA_HPP
#define DRAY_GRID_FUNCTION_DATA_HPP

#include <dray/array.hpp>
#include <dray/range.hpp>
#include <dray/vec.hpp>
#include <dray/data_model/dof_access.hpp>

#include <conduit.hpp>

namespace dray
{

template <int32 PhysDim> struct GridFunction
{
  // 1D flat map to elems, dofs
  Array<int32> m_ctrl_idx; // 0 <= ii < size_el, 0 <= jj < el_dofs, 0 <= m_ctrl_idx[ii*el_dofs + jj] < size_ctrl
  Array<Vec<Float, PhysDim>> m_values; // 0 <= kk < size_ctrl, 0 < c <= C, take m_values[kk][c].

  // number of dofs per element
  int32 m_el_dofs;
  // number of elements
  int32 m_size_el;
  // total length of the control index
  int32 m_size_ctrl;

  // zero copy into conduit node
  void to_node(conduit::Node &n_gf);

  void from_node(const conduit::Node &n_gf);

  void resize (int32 size_el, int32 el_dofs, int32 size_ctrl);
  void resize_counting (int32 size_el, int32 el_dofs);

  int32 get_num_elem () const
  {
    return m_size_el;
  }

  template <typename CoeffIterType>
  DRAY_EXEC static void get_elt_node_range (const CoeffIterType &coeff_iter,
                                            const int32 el_dofs,
                                            Range *comp_range);
};

// TODO: I dont think this function belongs here. it doesnt'
// even access anything
template <int32 PhysDim>
template <typename CoeffIterType>
DRAY_EXEC void GridFunction<PhysDim>::get_elt_node_range (const CoeffIterType &coeff_iter,
                                                         const int32 el_dofs,
                                                         Range *comp_range)
{
  // Assume that each component range is already initialized.
  for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
  {
    Vec<Float, PhysDim> node_val = coeff_iter[dof_idx];
    for (int32 pdim = 0; pdim < PhysDim; pdim++)
    {
      comp_range[pdim].include (node_val[pdim]);
    }
  }
}



//
// DeviceGridFunctionConst
//
template <int32 ncomp>
struct DeviceGridFunctionConst
{
  static constexpr int32 get_ncomp() { return ncomp; }

  const int32 * m_ctrl_idx_ptr;
  const Vec<Float, ncomp> * m_values_ptr;

  int32 m_el_dofs;
  int32 m_size_el;
  int32 m_size_ctrl;

  DeviceGridFunctionConst() = delete;
  DeviceGridFunctionConst(const GridFunction<ncomp> &gf);

  DRAY_EXEC ReadDofPtr<Vec<Float, ncomp>> get_rdp(int32 eidx) const;
};

template <int32 ncomp>
DeviceGridFunctionConst<ncomp>::DeviceGridFunctionConst(const GridFunction<ncomp> &gf)
  : m_ctrl_idx_ptr(gf.m_ctrl_idx.get_device_ptr_const()),
    m_values_ptr(gf.m_values.get_device_ptr_const()),
    m_el_dofs(gf.m_el_dofs),
    m_size_el(gf.m_size_el),
    m_size_ctrl(gf.m_size_ctrl)
  {}

template <int32 ncomp>
DRAY_EXEC ReadDofPtr<Vec<Float, ncomp>> DeviceGridFunctionConst<ncomp>::get_rdp(int32 eidx) const
{
  return { m_ctrl_idx_ptr + eidx * m_el_dofs, m_values_ptr };
}




//
// DeviceGridFunction
//
template <int32 ncomp>
struct DeviceGridFunction
{
  static constexpr int32 get_ncomp() { return ncomp; }

  const int32 * m_ctrl_idx_ptr;
  Vec<Float, ncomp> * m_values_ptr;

  int32 m_el_dofs;
  int32 m_size_el;
  int32 m_size_ctrl;

  DeviceGridFunction() = delete;
  DeviceGridFunction(GridFunction<ncomp> &gf);

  DRAY_EXEC WriteDofPtr<Vec<Float, ncomp>> get_wdp(int32 eidx) const;
  DRAY_EXEC ReadDofPtr<Vec<Float, ncomp>> get_rdp(int32 eidx) const;
};

template <int32 ncomp>
DeviceGridFunction<ncomp>::DeviceGridFunction(GridFunction<ncomp> &gf)
  : m_ctrl_idx_ptr(gf.m_ctrl_idx.get_device_ptr_const()),
    m_values_ptr(gf.m_values.get_device_ptr()),
    m_el_dofs(gf.m_el_dofs),
    m_size_el(gf.m_size_el),
    m_size_ctrl(gf.m_size_ctrl)
  {}

template <int32 ncomp>
DRAY_EXEC WriteDofPtr<Vec<Float, ncomp>> DeviceGridFunction<ncomp>::get_wdp(int32 eidx) const
{
  return { m_ctrl_idx_ptr + eidx * m_el_dofs, m_values_ptr };
}

template <int32 ncomp>
DRAY_EXEC ReadDofPtr<Vec<Float, ncomp>> DeviceGridFunction<ncomp>::get_rdp(int32 eidx) const
{
  return { m_ctrl_idx_ptr + eidx * m_el_dofs, m_values_ptr };
}




} // namespace dray
#endif // DRAY_GRID_FUNCTION_DATA_HPP
