// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/fragment.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const Fragment &frag)
{
  out << "[ scalar : " << frag.m_scalar << " norm: " << frag.m_normal << " ]";
  return out;
}

} // namespace dray
