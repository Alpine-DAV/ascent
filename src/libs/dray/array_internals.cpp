// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/array_internals.hpp>
#include <dray/types.hpp>

namespace dray
{
template class ArrayInternals<int32>;
template class ArrayInternals<int64>;
template class ArrayInternals<float32>;
template class ArrayInternals<float64>;
} // namespace dray
