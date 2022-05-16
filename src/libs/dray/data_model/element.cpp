// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/data_model/element.hpp>

namespace dray
{

//
// Explicit instantiations.

template class InvertibleElement_impl<3, ElemType::Tensor, Order::General>;
template class InvertibleElement_impl<3, ElemType::Simplex, Order::General>;
// If fixed-order implementations are needed as well, add instantiations for them here.

template class Element<2, 1, ElemType::Tensor, Order::General>;
template class Element<2, 3, ElemType::Tensor, Order::General>;
template class Element<3, 1, ElemType::Tensor, Order::General>;
template class Element<3, 3, ElemType::Tensor, Order::General>;
template class Element<2, 1, ElemType::Simplex, Order::General>;
template class Element<2, 3, ElemType::Simplex, Order::General>;
template class Element<3, 1, ElemType::Simplex, Order::General>;
template class Element<3, 3, ElemType::Simplex, Order::General>;

} // namespace dray
