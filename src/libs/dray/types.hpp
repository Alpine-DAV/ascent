// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TYPES_HPP
#define DRAY_TYPES_HPP

namespace dray
{

typedef unsigned char uint8;
typedef unsigned int uint32;
typedef unsigned long long int uint64;

typedef signed char int8;
typedef int int32;
typedef long long int int64;

typedef float float32;
typedef double float64;

#ifdef DRAY_COMBO_INT64
typedef int64 combo_int;
#else
typedef int32 combo_int;
#endif

#ifdef DRAY_DOUBLE_PRECISION
typedef double Float;
#else
typedef float Float;
#endif

}; // namespace dray
#endif
