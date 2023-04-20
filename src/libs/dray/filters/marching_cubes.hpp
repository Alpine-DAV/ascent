// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MARCHING_CUBES_HPP
#define DRAY_MARCHING_CUBES_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class MarchingCubes
{
  std::string m_field;
  std::vector<Float> m_isovalues;
public:
  MarchingCubes();
  ~MarchingCubes();

  void set_field(const std::string &name);
  void set_isovalue(Float value);
  void set_isovalues(const Float *values, int nvalues);

  Collection execute(Collection &);
};

}

#endif
