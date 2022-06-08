// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MESH_HPP
#define DRAY_MESH_HPP

#include <dray/array.hpp>
#include <dray/aabb.hpp>
#include <dray/location.hpp>
#include <conduit.hpp>
#include <string>

namespace dray
{

class Mesh
{
protected:
  std::string m_name;
  std::string m_shape_name;
public:
  virtual ~Mesh(){};

  std::string name() const { return m_name; }
  void name(const std::string &name) { m_name = name; }

  virtual std::string type_name() const = 0;
  virtual int32 cells() const = 0;
  virtual int32 order() const = 0;
  virtual int32 dims() const = 0;
  virtual AABB<3> bounds() = 0;
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) = 0;
  virtual void to_node(conduit::Node &n_topo) = 0;
};

} // namespace dray

#endif // DRAY_TOPOLGY_BASE_HPP
