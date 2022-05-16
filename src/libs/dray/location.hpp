// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LOCATION_HPP
#define DRAY_LOCATION_HPP

#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

class Location
{
  public:
  int32 m_cell_id;        /*!< Cell containing the location. -1 indicates not found */
  Vec<Float, 3> m_ref_pt; /*!< Refence space coordinates of location */
};

std::ostream &operator<< (std::ostream &out, const Location &loc);

} // namespace dray
#endif
