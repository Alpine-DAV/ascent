// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/volume_partial.hpp>
#include <iostream>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const VolumePartial &v)
{
  out<<"Id : "<<v.m_pixel_id<<" depth "<<v.m_depth<<" "<<v.m_color;
  return out;
}

} // namespace dray
