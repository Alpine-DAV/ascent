// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_PLANE_DETECTOR_HPP
#define DRAY_PLANE_DETECTOR_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class PlaneDetector
{
public:
  Vec<Float, 3> m_view = {{0.f, 0.f, 1.f}};
  Vec<Float, 3> m_up = {{0.f, 1.f, 0.f}};
  Vec<Float, 3> m_center = {{0.f, 0.f, 0.f}};
  Float m_plane_width = 1.f;
  Float m_plane_height = 1.f;
  int32 m_x_res = 1024;
  int32 m_y_res = 1024;
};

} // namespace dray
#endif
