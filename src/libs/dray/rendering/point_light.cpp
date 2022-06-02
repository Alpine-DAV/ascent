// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/point_light.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const PointLight &light)
{
  out << "{"<<light.m_pos<<" "<<light.m_amb<<" "<<light.m_diff<<" ";
  out << light.m_spec<<" "<<light.m_spec_pow<<"} ";
  return out;
}

} // namespace dray
