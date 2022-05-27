// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_COLORS_HPP
#define DRAY_COLORS_HPP

#include <dray/types.hpp>

namespace dray
{
using Color = Vec<float32,4>;

DRAY_EXEC void blend(Color &front, const Color &back)
{
  // composite
  const float32 alpha = back[3] * (1.f - front[3]);
  front[0] = front[0] + back[0] * alpha;
  front[1] = front[1] + back[1] * alpha;
  front[2] = front[2] + back[2] * alpha;
  front[3] = alpha + front[3];
}

DRAY_EXEC void blend_pre_alpha(Color &front, const Color &back)
{
  // composite
  const float32 alpha = (1.f - front[3]);
  front[0] = front[0] + back[0] * alpha;
  front[1] = front[1] + back[1] * alpha;
  front[2] = front[2] + back[2] * alpha;
  front[3] = front[3] + back[3] * alpha;
}

inline void blend_host(Color &front, const Color &back)
{
  // composite
  const float32 alpha = back[3] * (1.f - front[3]);
  front[0] = front[0] + back[0] * alpha;
  front[1] = front[1] + back[1] * alpha;
  front[2] = front[2] + back[2] * alpha;
  front[3] = alpha + front[3];
}

inline void pre_mult_alpha_blend_host(Color &front, const Color &back)
{
  // composite
  const float32 alpha = (1.f - front[3]);
  front[0] = front[0] + back[0] * alpha;
  front[1] = front[1] + back[1] * alpha;
  front[2] = front[2] + back[2] * alpha;
  front[3] = front[3] + back[3] * alpha;
}

static inline Color make_red()
{
  return {1.f, 0.f, 0.f, 1.f};
}

static inline Color make_green()
{
  return {0.f, 1.f, 0.f, 1.f};
}

static inline Color make_blue()
{
  return {0.f, 0.f, 1.f, 1.f};
}

static inline Color make_white()
{
  return {1.f, 1.f, 1.f, 1.f};
}

static inline Color make_clear()
{
  return {0.f, 0.f, 0.f, 0.f};
}

} // namespace dray
#endif
