// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SCALAR_BUFFER_HPP
#define DRAY_SCALAR_BUFFER_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

#include <conduit.hpp>
#include <map>

namespace dray
{

struct ScalarBuffer
{
  int32 m_width;
  int32 m_height;
  Float m_clear_value;

  std::map<std::string,Array<Float>> m_scalars;
  Array<Float> m_depths;
  Array<int32> m_zone_ids;

  ScalarBuffer();

  ScalarBuffer(const int32 width,
               const int32 height,
               const Float clear_value);

  bool has_field(const std::string name);
  void add_field(const std::string name);
  void to_node(conduit::Node &mesh);
  int32 size() const;
};

} // namespace dray
#endif
