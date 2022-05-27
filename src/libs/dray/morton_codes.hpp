// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MORTON_CODES_HPP
#define DRAY_MORTON_CODES_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/math.hpp>

namespace dray
{

// expands 10-bit unsigned int into 30 bits
DRAY_EXEC uint32 expand_bits32 (uint32 x32)
{
  x32 = (x32 | (x32 << 16)) & 0x030000FF;
  x32 = (x32 | (x32 << 8)) & 0x0300F00F;
  x32 = (x32 | (x32 << 4)) & 0x030C30C3;
  x32 = (x32 | (x32 << 2)) & 0x09249249;
  return x32;
}

DRAY_EXEC uint64 expand_bits64 (uint32 x)
{
  uint64 x64 = x & 0x1FFFFF;
  x64 = (x64 | x64 << 32) & 0x1F00000000FFFF;
  x64 = (x64 | x64 << 16) & 0x1F0000FF0000FF;
  x64 = (x64 | x64 << 8) & 0x100F00F00F00F00F;
  x64 = (x64 | x64 << 4) & 0x10c30c30c30c30c3;
  x64 = (x64 | x64 << 2) & 0x1249249249249249;

  return x64;
}

// Returns 30 bit morton code for coordinates for
// x, y, and z are expecting to be between [0,1]
DRAY_EXEC uint32 morton_3d (float32 &x, float32 &y, float32 &z)
{
  // take the first 10 bits
  x = fmin (fmax (x * 1024.0f, 0.0f), 1023.0f);
  y = fmin (fmax (y * 1024.0f, 0.0f), 1023.0f);
  z = fmin (fmax (z * 1024.0f, 0.0f), 1023.0f);
  // expand 10 bits to 30
  uint32 xx = expand_bits32 ((uint32)x);
  uint32 yy = expand_bits32 ((uint32)y);
  uint32 zz = expand_bits32 ((uint32)z);
  // interleave coordinates
  return (zz << 2 | yy << 1 | xx);
}

// Returns 30 bit morton code for coordinates for
// coordinates in the unit cude
DRAY_EXEC uint64 morton_3d_64 (float32 &x, float32 &y, float32 &z)
{
  // take the first 21 bits
  x = fmin (fmax (x * 2097152.0f, 0.0f), 2097151.0f);
  y = fmin (fmax (y * 2097152.0f, 0.0f), 2097151.0f);
  z = fmin (fmax (z * 2097152.0f, 0.0f), 2097151.0f);
  // expand the 10 bits to 30
  uint64 xx = expand_bits64 ((uint32)x);
  uint64 yy = expand_bits64 ((uint32)y);
  uint64 zz = expand_bits64 ((uint32)z);
  // interleave coordinates
  return (zz << 2 | yy << 1 | xx);
}

} // namespace dray
#endif
