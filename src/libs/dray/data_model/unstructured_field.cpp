// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_model/field.hpp>
#include <dray/data_model/device_field.hpp>
#include <dray/policies.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>

#include <dray/data_model/element.hpp>
#include <dray/array_utils.hpp>


namespace dray
{

namespace detail
{

DRAY_EXEC
void convert_ref_pt(const Vec<Float,3> &src_pt, Vec<Float,2> &dest_pt)
{
  dest_pt[0] = src_pt[0];
  dest_pt[1] = src_pt[1];
}

DRAY_EXEC
void convert_ref_pt(const Vec<Float,3> &src_pt, Vec<Float,3> &dest_pt)
{
  dest_pt = src_pt;
}

template <class ElemT> std::vector<Range> get_range (const UnstructuredField<ElemT> &field)
{

  RAJA::ReduceMin<reduce_policy, Float> comp_xmin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> comp_xmax (neg_infinity<Float>());

  RAJA::ReduceMin<reduce_policy, Float> comp_ymin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> comp_ymax (neg_infinity<Float>());

  RAJA::ReduceMin<reduce_policy, Float> comp_zmin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> comp_zmax (neg_infinity<Float>());

  const int32 num_nodes = field.get_dof_data ().m_values.size ();
  const int32 entries = num_nodes / ElemT::get_ncomp();
  constexpr int32 comps = ElemT::get_ncomp();
  assert(comps < 4);
  if(comps > 3)
  {
    DRAY_ERROR("We didn't plan for "<<comps<<" components");
  }

  const Vec<Float,comps> *node_val_ptr =
    field.get_dof_data ().m_values.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, entries), [=] DRAY_LAMBDA (int32 ii) {

    // this has to be inside the lambda for gcc8.1 otherwise:
    // error: use of 'this' in a constant expression
    // so we add another definition
    constexpr int32 lmbd_comps = ElemT::get_ncomp();
    const Vec<Float,lmbd_comps> value = node_val_ptr[ii];

    if(comps > 0)
    {
      comp_xmin.min (value[0]);
      comp_xmax.max (value[0]);
    }
    if(comps > 1)
    {
      comp_ymin.min (value[1]);
      comp_ymax.max (value[1]);
    }
    if(comps > 2)
    {
      comp_zmin.min (value[2]);
      comp_zmax.max (value[2]);
    }
  });
  DRAY_ERROR_CHECK();

  std::vector<Range> ranges;
  if(comps > 0)
  {
    Range range;
    if(num_nodes > 0)
    {
      range.include (comp_xmin.get ());
      range.include (comp_xmax.get ());
    }
    ranges.push_back(range);
  }
  if(comps > 1)
  {
    Range range;
    if(num_nodes > 0)
    {
      range.include (comp_ymin.get ());
      range.include (comp_ymax.get ());
    }
    ranges.push_back(range);
  }
  if(comps > 2)
  {
    Range range;
    if(num_nodes > 0)
    {
      range.include (comp_zmin.get ());
      range.include (comp_zmax.get ());
    }
    ranges.push_back(range);
  }
  //std::cout<<"num_nodes "<<num_nodes<<" "<<ranges[0]<<" "<<comps<<"\n";
  return ranges;
}

} // namespace detail

template <class ElemT>
UnstructuredField<ElemT>::UnstructuredField (const GridFunction<ElemT::get_ncomp ()> &dof_data,
                     int32 poly_order,
                     const std::string name)
: m_dof_data (dof_data),
  m_poly_order (poly_order),
  m_range_calculated(false)
{
  this->name(name);

  // check to see if this is a valid construction
  if(ElemT::get_P() != Order::General)
  {
    bool valid = false;
    if(m_poly_order == 0 && ElemT::get_P() == Order::Constant)
    {
      valid= true;
    }
    else if(m_poly_order == 1 && ElemT::get_P() == Order::Linear)
    {
      valid= true;
    }
    else if(m_poly_order == 2 && ElemT::get_P() == Order::Quadratic)
    {
      valid = true;
    }

    if(!valid)
    {
      DRAY_ERROR("Fixed order mismatch. Poly order "<<m_poly_order
                   <<" template "<<ElemT::get_P());
    }
  }
}


template <class ElemT>
UnstructuredField<ElemT>::UnstructuredField(const UnstructuredField &other)
  : m_dof_data(other.m_dof_data),
    m_poly_order(other.m_poly_order),
    m_range_calculated(other.m_range_calculated),
    m_ranges(other.m_ranges)
{
  this->name(other.name());
}

template <class ElemT>
UnstructuredField<ElemT>::UnstructuredField(UnstructuredField &&other)
  : m_dof_data(other.m_dof_data),
    m_poly_order(other.m_poly_order),
    m_range_calculated(other.m_range_calculated),
    m_ranges(other.m_ranges)
{
  this->name(other.name());
}

template<typename ElemT>
void UnstructuredField<ElemT>::to_node(conduit::Node &n_field)
{
  n_field.reset();
  n_field["type_name"] = type_name();
  n_field["order"] = get_poly_order();

  conduit::Node &n_gf = n_field["grid_function"];
  GridFunction<ElemT::get_ncomp ()> gf = get_dof_data();
  gf.to_node(n_gf);

}

template <class ElemT> std::vector<Range> UnstructuredField<ElemT>::range () const
{
  if(!m_range_calculated)
  {
    m_ranges = detail::get_range (*this);
    m_range_calculated = true;
  }
  return m_ranges;
}

template <class ElemT>
int32 UnstructuredField<ElemT>::order() const
{
  return m_poly_order;
}

template <class ElemT>
std::string UnstructuredField<ElemT>::type_name() const
{
  return element_name<ElemT>(ElemT());
}

template <class ElemT>
void UnstructuredField<ElemT>::eval(const Array<Location> locs, Array<Float> &values)
{

  const int32 size = locs.size();
  // allow people to pass in values
  if(values.size() != size)
  {
    values.resize(size);
  }

  DeviceField<ElemT> d_field(*this);

  const Location *locs_ptr = locs.get_device_ptr_const();
  Float * values_ptr = values.get_device_ptr();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    const Location loc = locs_ptr[ii];
    if(loc.m_cell_id != -1)
    {
      ElemT elem = d_field.get_elem(loc.m_cell_id);
      Vec<Float, ElemT::get_dim()> ref_pt;
      detail::convert_ref_pt(loc.m_ref_pt, ref_pt);
      Vec<Float,ElemT::get_ncomp()> val = elem.eval(ref_pt);
      // TODO: should change the array class to be multi-component
      // that is, we don't really have a vector path right now, so
      // we only support scalars currently
      // Thus, just shove the scalar into values
      values_ptr[ii] = val[0];
    }
  });

}

template <class ElemT>
UnstructuredField<ElemT> UnstructuredField<ElemT>::uniform_field(int32 num_els,
                                         const Vec<Float, ElemT::get_ncomp()> &val,
                                         const std::string &name)
{
  const auto shape = adapt_get_shape(ElemT());
  const auto order_p = adapt_get_order_policy(ElemT(), 0);
  const int32 order = eattr::get_order(order_p);
  const int32 npe = eattr::get_num_dofs(shape, order_p);
  constexpr int32 ncomp = ElemT::get_ncomp();
  GridFunction<ncomp> gf;
  gf.resize(num_els, npe, 1);
  gf.m_values.get_host_ptr()[0] = val;
  array_memset(gf.m_ctrl_idx, 0);

  return UnstructuredField(gf, order, name);
}


// Explicit instantiations.
template class UnstructuredField<HexScalar>;
template class UnstructuredField<HexScalar_P0>;
template class UnstructuredField<HexScalar_P1>;
template class UnstructuredField<HexScalar_P2>;

template class UnstructuredField<TetScalar>;
template class UnstructuredField<TetScalar_P0>;
template class UnstructuredField<TetScalar_P1>;
template class UnstructuredField<TetScalar_P2>;

template class UnstructuredField<QuadScalar>;
template class UnstructuredField<QuadScalar_P0>;
template class UnstructuredField<QuadScalar_P1>;
template class UnstructuredField<QuadScalar_P2>;

template class UnstructuredField<TriScalar>;
template class UnstructuredField<TriScalar_P0>;
template class UnstructuredField<TriScalar_P1>;
template class UnstructuredField<TriScalar_P2>;

template class UnstructuredField<HexVector>;
template class UnstructuredField<HexVector_P0>;
template class UnstructuredField<HexVector_P1>;
template class UnstructuredField<HexVector_P2>;

template class UnstructuredField<QuadVector>;
template class UnstructuredField<QuadVector_P0>;
template class UnstructuredField<QuadVector_P1>;
template class UnstructuredField<QuadVector_P2>;

template class UnstructuredField<QuadVector_2D>;
template class UnstructuredField<QuadVector_2D_P0>;
template class UnstructuredField<QuadVector_2D_P1>;
template class UnstructuredField<QuadVector_2D_P2>;

template class UnstructuredField<TetVector>;
template class UnstructuredField<TetVector_P0>;
template class UnstructuredField<TetVector_P1>;
template class UnstructuredField<TetVector_P2>;

template class UnstructuredField<TriVector>;
template class UnstructuredField<TriVector_P0>;
template class UnstructuredField<TriVector_P1>;
template class UnstructuredField<TriVector_P2>;

template class UnstructuredField<TriVector_2D>;
template class UnstructuredField<TriVector_2D_P0>;
template class UnstructuredField<TriVector_2D_P1>;
template class UnstructuredField<TriVector_2D_P2>;



} // namespace dray
