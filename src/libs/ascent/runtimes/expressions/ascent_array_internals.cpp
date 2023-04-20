// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "ascent_array_internals.hpp"

namespace ascent
{
namespace runtime
{

template class ArrayInternals<unsigned char>;
template class ArrayInternals<int>;
template class ArrayInternals<long long int>;
template class ArrayInternals<float>;
template class ArrayInternals<double>;

} // namespace runtime
} // namespace ascent
