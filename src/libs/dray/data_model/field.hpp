// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FIELD_HPP
#define DRAY_FIELD_HPP

#include <string>
#include <vector>
#include <conduit.hpp>
#include <dray/array.hpp>
#include <dray/location.hpp>
#include <dray/range.hpp>

namespace dray
{

class Field
{
protected:
  std::string m_name;
  // each field is associated with one mesh
  std::string m_mesh;
public:
  virtual ~Field() {}

  std::string name() const
  {
    return m_name;
  }

  void name(const std::string &name)
  {
    m_name = name;
  }

  void mesh_name(const std::string &name)
  {
    m_mesh = name;
  }

  std::string mesh_name() const
  {
    return m_mesh;
  }

  virtual std::vector<Range> range() const = 0;
  virtual int32 order() const = 0;
  virtual int32 components() const = 0;
  virtual std::string type_name() const = 0;
  virtual void to_node(conduit::Node &n_field) = 0;
  virtual void eval(const Array<Location> locs, Array<Float> &values) = 0;
};

} // namespace dray

#endif // DRAY_FIELD_BASE_HPP
