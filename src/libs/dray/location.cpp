// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/location.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const Location &loc)
{
  out << "[ cell_id " << loc.m_cell_id << " ref_pt: " << loc.m_ref_pt << " ]";
  return out;
}

} // namespace dray
