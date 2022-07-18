// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <RAJA/RAJA.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/mesh_utils.hpp>
#include <dray/aabb.hpp>
#include <dray/error_check.hpp>
#include <dray/array_utils.hpp>
#include <dray/dray.hpp>
#include <dray/policies.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/data_model/element.hpp>


namespace dray
{

template <class Element> const BVH UnstructuredMesh<Element>::get_bvh ()
{
  if(!m_is_constructed)
  {
    m_bvh = detail::construct_bvh (*this, m_ref_aabbs);
    m_is_constructed = true;
  }
  return m_bvh;
}

template <class Element>
UnstructuredMesh<Element>::UnstructuredMesh (const GridFunction<3u> &dof_data, int32 poly_order)
: m_dof_data (dof_data),
  m_poly_order (poly_order),
  m_is_constructed(false)
{
  // check to see if this is a valid construction
  if(Element::get_P() != Order::General)
  {
    bool valid = false;
    if(m_poly_order == 1 && Element::get_P() == Order::Linear)
    {
      valid= true;
    }
    else if(m_poly_order == 2 && Element::get_P() == Order::Quadratic)
    {
      valid = true;
    }

    if(!valid)
    {
      DRAY_ERROR("Fixed order mismatch. Poly order "<<m_poly_order
                   <<" template "<<Element::get_P());
    }
  }
}

template <class Element>
UnstructuredMesh<Element>::~UnstructuredMesh()
{
}

template <class Element>
UnstructuredMesh<Element>::UnstructuredMesh(const UnstructuredMesh &other)
  : m_dof_data(other.m_dof_data),
    m_poly_order(other.m_poly_order),
    m_is_constructed(other.m_is_constructed),
    m_bvh(other.m_bvh),
    m_ref_aabbs(other.m_ref_aabbs)
{
  // check to see if this is a valid construction
  if(Element::get_P() != Order::General)
  {
    bool valid = false;
    if(m_poly_order == 1 && Element::get_P() == Order::Linear)
    {
      valid= true;
    }
    else if(m_poly_order == 2 && Element::get_P() == Order::Quadratic)
    {
      valid = true;
    }

    if(!valid)
    {
      DRAY_ERROR("Fixed order mismatch. Poly order "<<m_poly_order
                   <<" template "<<Element::get_P());
    }
  }
}

template <class Element>
UnstructuredMesh<Element>::UnstructuredMesh(UnstructuredMesh &&other)
  : m_dof_data(other.m_dof_data),
    m_poly_order(other.m_poly_order),
    m_is_constructed(other.m_is_constructed),
    m_bvh(other.m_bvh),
    m_ref_aabbs(other.m_ref_aabbs)
{
  // check to see if this is a valid construction
  if(Element::get_P() != Order::General)
  {
    bool valid = false;
    if(m_poly_order == 1 && Element::get_P() == Order::Linear)
    {
      valid= true;
    }
    else if(m_poly_order == 2 && Element::get_P() == Order::Quadratic)
    {
      valid = true;
    }

    if(!valid)
    {
      DRAY_ERROR("Fixed order mismatch. Poly order "<<m_poly_order
                   <<" template "<<Element::get_P());
    }
  }
}

template <class Element>
Array<Location> UnstructuredMesh<Element>::locate (Array<Vec<Float, 3u>> &wpoints)
{
  DRAY_LOG_OPEN ("locate");

  const int32 size = wpoints.size ();
  Array<Location> locations;
  locations.resize (size);

  Location *loc_ptr = locations.get_device_ptr ();
  const Vec<Float,3> *points_ptr = wpoints.get_device_ptr_const();

  DeviceMesh<Element> device_mesh (*this);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {

    Location loc = { -1, { -1.f, -1.f, -1.f } };
    const Vec<Float, 3> target_pt = points_ptr[i];
    loc_ptr[i] = device_mesh.locate(target_pt);
  });

  DRAY_ERROR_CHECK();
  DRAY_LOG_CLOSE();

  return locations;
}

template<typename Element>
int32 UnstructuredMesh<Element>::cells() const
{
  return m_dof_data.get_num_elem ();
}

template<typename Element>
int32 UnstructuredMesh<Element>::order() const
{
  return m_poly_order;
}

template<typename Element>
AABB<3> UnstructuredMesh<Element>::bounds()
{
  return get_bvh().m_bounds;
}

template<typename Element>
int32 UnstructuredMesh<Element>::dims() const
{
  return dim;
}

template<typename Element>
std::string UnstructuredMesh<Element>::type_name() const
{
  return element_name<Element>();
}

template<typename Element>
void UnstructuredMesh<Element>::to_node(conduit::Node &n_topo)
{
  n_topo.reset();
  n_topo["type_name"] = type_name();
  n_topo["order"] = order();

  conduit::Node &n_gf = n_topo["grid_function"];
  m_dof_data.to_node(n_gf);

}

// Currently supported topologies
template class UnstructuredMesh<Hex3>;
template class UnstructuredMesh<Hex_P1>;
template class UnstructuredMesh<Hex_P2>;

template class UnstructuredMesh<Tet3>;
template class UnstructuredMesh<Tet_P1>;
template class UnstructuredMesh<Tet_P2>;

template class UnstructuredMesh<Quad3>;
template class UnstructuredMesh<Quad_P1>;
template class UnstructuredMesh<Quad_P2>;

template class UnstructuredMesh<Tri3>;
template class UnstructuredMesh<Tri_P1>;
template class UnstructuredMesh<Tri_P2>;

} // namespace dray
