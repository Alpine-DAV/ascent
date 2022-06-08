// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_VOLUME_PARTIAL_HPP
#define DRAY_VOLUME_PARTIAL_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <iostream>

namespace dray
{

struct VolumePartial
{
  int32 m_pixel_id;
  float32 m_depth;
  Vec<float32,4> m_color;
};

std::ostream &operator<< (std::ostream &out, const VolumePartial &v);

} // namespace dray
#endif
