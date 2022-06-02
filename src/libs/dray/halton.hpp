// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_HALTON_HPP
#define DRAY_HALTON_HPP

#include <dray/vec.hpp>

namespace dray
{

template <typename T, int32 Base>
static DRAY_EXEC void Halton2D (const int32 &sampleNum, Vec<T, 2> &coord)
{
  // generate base2 halton (use bit arithmetic)
  T x = 0.0f;
  T xadd = 1.0f;
  uint32 b2 = 1 + sampleNum;
  while (b2 != 0)
  {
    xadd *= 0.5f;
    if ((b2 & 1) != 0) x += xadd;
    b2 >>= 1;
  }

  // generate arbitrary Base Halton
  T y = 0.0f;
  T yadd = 1.0f;
  int32 bn = 1 + sampleNum;
  while (bn != 0)
  {
    yadd *= 1.0f / (T)Base;
    y += (T) (bn % Base) * yadd;
    bn /= Base;
  }

  coord[0] = x;
  coord[1] = y;
} // Halton2D

} // namespace dray

#endif
