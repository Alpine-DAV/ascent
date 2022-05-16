// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RANDOM_HPP
#define DRAY_RANDOM_HPP

#include <dray/array.hpp>
#include <dray/types.hpp>
#include <dray/exports.hpp>

#include <stdlib.h>
#include <time.h>

namespace dray
{
namespace detail
{
union UIntFloat
{
  uint32 U;
  float32 F;
};
} //  namespace detail
//https://github.com/opencv/opencv/blob/85ade61ef7d95fcca19cc6d3eba532225d7790a2/modules/core/include/opencv2/core/operations.hpp#L374
// multiple with cary random numbers
DRAY_EXEC
static uint32 random(Vec<uint32,2> &rng_state)
{
  rng_state[0] = 36969 * (rng_state[0] & 65535) + (rng_state[0] >> 16);
  rng_state[1] = 18000 * (rng_state[1] & 65535) + (rng_state[1] >> 16);

  uint32 res  = (rng_state[0] << 16) + rng_state[1];
  return res;
}

DRAY_EXEC
static float32 randomf(Vec<uint32,2> &rng_state)
{
  detail::UIntFloat r;
  uint32 res  = random(rng_state);
  r.U = (res & 0x007fffff) | 0x40000000;
  return (r.F - 2.f) / 2.f;
}

DRAY_EXEC
static void seed_rng(Array<Vec<uint32,2>> &rng_states,
                     bool deterministic = false)
{
  const int32 size = rng_states.size();
  Vec<uint32,2> *state_ptr = rng_states.get_host_ptr();
  if(deterministic)
  {
    srand (0);
  }
  else
  {
    srand ((uint32(time(NULL))));
  }

  for(int i = 0; i < size; ++i)
  {
    Vec<uint32, 2> state;
    state[0] = static_cast<uint32>(rand());
    state[1] = static_cast<uint32>(rand());
    state_ptr[i] = state;
  }
}
} // namespace dray
#endif
