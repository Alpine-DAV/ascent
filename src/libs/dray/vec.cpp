// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/vec.hpp>

namespace dray
{
template class Vec<int32, 1>;
template class Vec<uint32, 1>;
template class Vec<int64, 1>;
template class Vec<float32, 1>;
template class Vec<float64, 1>;

template class Vec<uint8, 2>;
template class Vec<int32, 2>;
template class Vec<uint32, 2>;
template class Vec<int64, 2>;
template class Vec<float32, 2>;
template class Vec<float64, 2>;

template class Vec<uint8, 3>;
template class Vec<uint32, 3>;
template class Vec<int32, 3>;
template class Vec<int64, 3>;
template class Vec<float32, 3>;
template class Vec<float64, 3>;

template class Vec<uint32, 4>;
template class Vec<int32, 4>;
template class Vec<int64, 4>;
template class Vec<float32, 4>;
template class Vec<float64, 4>;
} // namespace dray
