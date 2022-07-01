// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_REF_POINT_HPP
#define DRAY_REF_POINT_HPP

#include <dray/vec.hpp>

namespace dray
{
template <int32 dim = 3> struct RefPoint
{
  int32 m_el_id;
  Vec<Float, dim> m_el_coords;
};

template <int32 dim>
std::ostream &operator<< (std::ostream &out, const RefPoint<dim> &rpt)
{
  out << rpt.m_el_id;
  return out;
}

} // namespace dray

#endif // DRAY_REF_POINT_HPP
