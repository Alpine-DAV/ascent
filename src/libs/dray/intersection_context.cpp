// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/intersection_context.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const IntersectionContext &r)
{
  out << r.m_pixel_id;
  return out;
}

} // namespace dray
