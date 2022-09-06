// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include <dray/data_model/elem_attr.hpp>

TEST (dray_elem_attr, dray_elem_attr)
{
  using dray::General;

  std::cout << "Linear" << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeTri{}, dray::OrderPolicy<1>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeTri{}, dray::OrderPolicy<General>{1}) << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeQuad{}, dray::OrderPolicy<1>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeQuad{}, dray::OrderPolicy<General>{1}) << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeTet{}, dray::OrderPolicy<1>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeTet{}, dray::OrderPolicy<General>{1}) << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeHex{}, dray::OrderPolicy<1>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeHex{}, dray::OrderPolicy<General>{1}) << "\n";

  std::cout << "\n";

  std::cout << "Quadratic" << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeTri{}, dray::OrderPolicy<2>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeTri{}, dray::OrderPolicy<General>{2}) << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeQuad{}, dray::OrderPolicy<2>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeQuad{}, dray::OrderPolicy<General>{2}) << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeTet{}, dray::OrderPolicy<2>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeTet{}, dray::OrderPolicy<General>{2}) << "\n";

  std::cout << dray::eattr::get_num_dofs(dray::ShapeHex{}, dray::OrderPolicy<2>{}) << "\n";
  std::cout << dray::eattr::get_num_dofs(dray::ShapeHex{}, dray::OrderPolicy<General>{2}) << "\n";

}
