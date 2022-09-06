// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MATERIAL_HPP
#define DRAY_MATERIAL_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

struct Material
{
  int32 m_id;
  Vec<float32,3> m_diffuse;
  Vec<float32,3> m_ambient;
  Vec<float32,3> m_specular;
  int32 m_diff_texture;
};

std::ostream &operator<< (std::ostream &out, const Material &mat);

} // namespace dray
#endif
