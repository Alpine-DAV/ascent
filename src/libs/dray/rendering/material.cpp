// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <dray/rendering/material.hpp>

namespace dray
{

std::ostream &operator<< (std::ostream &out, const Material &mat)
{
  out << "mat id "<<mat.m_id
      <<" {"<<mat.m_ambient<<" "
      <<mat.m_diffuse<<" "
      <<mat.m_specular<<" ";
  out << "diff_text_id : "<<mat.m_diff_texture<<" } ";
  return out;
}

} // namespace dray
