// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef ASCENT_MATH_HPP
#define ASCENT_MATH_HPP

#include "ascent_execution_policies.hpp"

// include math so we can use functions defined
// in both cuda and c
#include <math.h>

#define ASCENT_INF_32 0x7f800000U
#define ASCENT_NG_INF_32 0xff800000U

#define ASCENT_INF_64 0x7ff0000000000000ULL
#define ASCENT_NG_INF_64 0xfff0000000000000ULL

#define ASCENT_NAN_32 0x7FC00000U
#define ASCENT_NAN_64 0x7FF8000000000000ULL

#define ASCENT_EPSILON_32 1e-4f
#define ASCENT_EPSILON_64 1e-8f

#ifndef __CUDACC__
// make sure min / max resolve for both cuda and cpu
#include <algorithm>
#include <math.h>
#include <string.h> //resolve memcpy
using namespace std;
#endif

namespace ascent
{

typedef unsigned char uint8;
typedef unsigned int uint32;
typedef unsigned long long int uint64;

typedef char int8;
typedef int int32;
typedef long long int int64;

namespace detail
{

union Bits32 {
  float scalar;
  uint32 bits;
};

union Bits64 {
  double scalar;
  uint64 bits;
};

} // namespace detail

template <typename T> ASCENT_EXEC T epsilon ()
{
  return 1;
}

template <> ASCENT_EXEC float epsilon<float> ()
{
  return ASCENT_EPSILON_32;
}

template <> ASCENT_EXEC double epsilon<double> ()
{
  return ASCENT_EPSILON_64;
}

ASCENT_EXEC
float nan32 ()
{
  detail::Bits32 nan;
  nan.bits = ASCENT_NAN_32;
  return nan.scalar;
}

ASCENT_EXEC
float infinity32 ()
{
  detail::Bits32 inf;
  inf.bits = ASCENT_INF_32;
  return inf.scalar;
}

ASCENT_EXEC
float neg_infinity32 ()
{
  detail::Bits32 ninf;
  ninf.bits = ASCENT_NG_INF_32;
  return ninf.scalar;
}

ASCENT_EXEC
double nan64 ()
{
  detail::Bits64 nan;
  nan.bits = ASCENT_NAN_64;
  return nan.scalar;
}

ASCENT_EXEC
double infinity64 ()
{
  detail::Bits64 inf;
  inf.bits = ASCENT_INF_64;
  return inf.scalar;
}

ASCENT_EXEC
double neg_infinity64 ()
{
  detail::Bits64 ninf;
  ninf.bits = ASCENT_NG_INF_64;
  return ninf.scalar;
}

template <typename T> ASCENT_EXEC T infinity ();

template <> ASCENT_EXEC float infinity<float> ()
{
  return infinity32 ();
}

template <> ASCENT_EXEC double infinity<double> ()
{
  return infinity64 ();
}

template <typename T> ASCENT_EXEC T nan();

template <> ASCENT_EXEC float nan<float> ()
{
  return nan32 ();
}

template <> ASCENT_EXEC double nan<double> ()
{
  return nan64 ();
}

template <typename T> ASCENT_EXEC T neg_infinity ();

template <> ASCENT_EXEC float neg_infinity<float> ()
{
  return neg_infinity32 ();
}

template <> ASCENT_EXEC double neg_infinity<double> ()
{
  return neg_infinity64 ();
}

//
// count leading zeros
//
ASCENT_EXEC
int32 clz (uint32 x)
{
  uint32 y;
  uint32 n = 32;
  y = x >> 16;
  if (y != 0)
  {
    n = n - 16;
    x = y;
  }
  y = x >> 8;
  if (y != 0)
  {
    n = n - 8;
    x = y;
  }
  y = x >> 4;
  if (y != 0)
  {
    n = n - 4;
    x = y;
  }
  y = x >> 2;
  if (y != 0)
  {
    n = n - 2;
    x = y;
  }
  y = x >> 1;
  if (y != 0) return int32 (n - 2);
  return int32 (n - x);
}

ASCENT_EXEC
double pi ()
{
  return 3.14159265358979323846264338327950288;
}

ASCENT_EXEC
float rcp (float f)
{
  return 1.0f / f;
}

ASCENT_EXEC
bool is_inf(const double f)
{
  return (2*f == f) && (f != 0);
}

ASCENT_EXEC
bool is_inf(const float f)
{
  return (2*f == f) && (f != 0);
}

ASCENT_EXEC
double rcp (double f)
{
  return 1.0 / f;
}

ASCENT_EXEC
double rcp_safe (double f)
{
  return rcp ((fabs (f) < 1e-8) ? (signbit (f) ? -1e-8 : 1e-8) : f);
}

ASCENT_EXEC
float rcp_safe (float f)
{
  return rcp ((fabs (f) < 1e-8f) ? (signbit (f) ? -1e-8f : 1e-8f) : f);
}

ASCENT_EXEC
float lerp(const float &t0, const float &t1, const float &t)
{
  return (t0 + t * (t1 - t0));
}



template <typename T>
ASCENT_EXEC T clamp (const T &val, const T &min_val, const T &max_val)
{
  return min (max_val, max (min_val, val));
}

// clamped hermite interpolation
ASCENT_EXEC
float smoothstep(const float e0, const float e1, float x) {
  x = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
  return x * x * (3 - 2 * x);
}

static constexpr ASCENT_EXEC float pi_180f ()
{
  return 0.01745329251994329547437168059786927f;
}
static constexpr ASCENT_EXEC double pi_180 ()
{
  return 0.01745329251994329547437168059786927;
}


} // namespace ascent
#endif
