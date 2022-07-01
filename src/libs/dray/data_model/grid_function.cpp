// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_model/grid_function.hpp>
#include <dray/array_utils.hpp>
#include <type_traits>

namespace dray
{

template <int32 PhysDim>
void GridFunction<PhysDim>::to_node(conduit::Node &n_gf)
{
  n_gf["dofs_per_element"] = m_el_dofs;
  n_gf["num_elemements"] = m_size_el;
  n_gf["values_size"] = m_values.size();
  n_gf["conn_size"] = m_size_ctrl;
  n_gf["phys_dim"] = PhysDim;

  Vec<Float,PhysDim> *values_ptr = m_values.get_host_ptr();
  Float *values_float_ptr = (Float*)(&(values_ptr[0][0]));
  n_gf["values"].set_external(values_float_ptr, m_values.size() * PhysDim);

  int32 *conn_ptr = m_ctrl_idx.get_host_ptr();
  n_gf["conn"].set_external(conn_ptr, m_ctrl_idx.size());
}

template <int32 PhysDim>
void GridFunction<PhysDim>::from_node(const conduit::Node &n_gf)
{
  if(n_gf["phys_dim"].to_int32() != PhysDim)
  {
    std::cout<<"node dim "<<n_gf["phys_dim"].to_int32()<<" phys dim "<<PhysDim<<"\n";
    std::cout<<"Mismatched phys dims\n";
  }

  int32 el_dofs = n_gf["dofs_per_element"].to_int32();
  int32 size_el = n_gf["num_elemements"].to_int32();
  int32 size_ctrl = n_gf["conn_size"].to_int32();

  resize(size_el, el_dofs, size_ctrl);

  // if we have data copy it, otherwise just leave the
  // memory allccated but not filled
  if(n_gf.has_path("values") && !n_gf["values"].dtype().is_empty())
  {
    const int32 vsize = n_gf["values"].dtype().number_of_elements();

    if(m_size_ctrl != vsize / PhysDim)
    {
      std::cout<<"Error: mismatched values size\n";
    }

    Vec<Float,PhysDim> *values_ptr = m_values.get_host_ptr();
    Float *values_float_ptr = (Float*)(&(values_ptr[0][0]));

    if(std::is_same<float32,Float>::value)
    {
      const Vec<Float,PhysDim> *in_values
        = (const Vec<Float,PhysDim>*)n_gf["values"].as_float32_ptr();
      m_values.set(in_values, m_size_ctrl);
    }
    else
    {
      const Vec<Float,PhysDim> *in_values
        = (const Vec<Float,PhysDim>*)n_gf["values"].as_float64_ptr();
      m_values.set(in_values, m_size_ctrl);
    }
  }

  if(n_gf.has_path("conn") && !n_gf["conn"].dtype().is_empty())
  {
    const int32 csize = n_gf["conn"].dtype().number_of_elements();
    if(m_ctrl_idx.size() != csize)
    {
      std::cout<<"Error: mismatched conn size\n";
    }
    const int32 *in_conn = n_gf["conn"].as_int32_ptr();
    m_ctrl_idx.set(in_conn, csize);
  }
}

template <int32 PhysDim>
void GridFunction<PhysDim>::resize (int32 size_el, int32 el_dofs, int32 size_ctrl)
{
  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_ctrl;

  m_ctrl_idx.resize (size_el * el_dofs);
  m_values.resize (size_ctrl);
}

template <int32 PhysDim>
void GridFunction<PhysDim>::resize_counting (int32 size_el, int32 el_dofs)
{
  m_el_dofs = el_dofs;
  m_size_el = size_el;
  m_size_ctrl = size_el * el_dofs;

  m_ctrl_idx = array_counting(size_el * el_dofs, 0, 1);
  m_values.resize (size_el * el_dofs);
}


template struct GridFunction<3>;
template struct GridFunction<2>;
template struct GridFunction<1>;

} // namespace dray
