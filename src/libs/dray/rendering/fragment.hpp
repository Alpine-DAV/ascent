// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FRAGMENT_HPP
#define DRAY_FRAGMENT_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class Fragment
{
  public:
  float32 m_scalar; /*!< non-normalized scalar value */
  Vec<float32, 3> m_normal; /*!< non-normalized surface normal or scalar gradient */
};

std::ostream &operator<< (std::ostream &out, const Fragment &frag);

} // namespace dray
#endif
