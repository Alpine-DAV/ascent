// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/matrix.hpp>

namespace dray
{
template class Matrix<float32, 4, 4>;
template class Matrix<float64, 4, 4>;
template class Matrix<float32, 3, 3>;
template class Matrix<float64, 3, 3>;

template class Matrix<float32, 4, 3>;
template class Matrix<float64, 4, 3>;
template class Matrix<float32, 4, 1>;
template class Matrix<float64, 4, 1>;
template class Matrix<float32, 3, 1>;
template class Matrix<float64, 3, 1>;
template class Matrix<float32, 1, 3>;
template class Matrix<float64, 1, 3>;
} // namespace dray
