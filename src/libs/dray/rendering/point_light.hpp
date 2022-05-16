// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POINT_LIGHT_HPP
#define DRAY_POINT_LIGHT_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

struct PointLight
{
  Vec<float32, 3> m_pos = {{0.f, 0.f, 0.f}};
  Vec<float32, 3> m_amb = {{0.4f, 0.4f, 0.4f}};
  Vec<float32, 3> m_diff = {{0.75f, 0.75f, 0.75f}};
  Vec<float32, 3> m_spec = {{0.8f, 0.8f, 0.8f}};
  //
  //Vec<float32, 3> m_pos = {{0.f, 0.f, 0.f}};
  //Vec<float32, 3> m_amb = {{0.0f, 0.0f, 0.0f}};
  //Vec<float32, 3> m_diff = {{0.75f, 0.75f, 0.75f}};
  //Vec<float32, 3> m_spec = {{0.0f, 0.0f, 0.0f}};

  //Vec<float32, 3> m_pos = {{0.f, 0.f, 0.f}};
  //Vec<float32, 3> m_amb = {{0.0f, 0.0f, 0.0f}};
  //Vec<float32, 3> m_diff = {{0.0f, 0.0f, 0.0f}};
  //Vec<float32, 3> m_spec = {{0.8f, 0.0f, 0.0f}};

  float32 m_spec_pow = 15.f;
};

std::ostream &operator<< (std::ostream &out, const PointLight &light);

} // namespace dray
#endif
