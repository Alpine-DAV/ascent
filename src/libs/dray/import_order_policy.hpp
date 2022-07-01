// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_IMPORT_ORDER_POLICY_HPP
#define DRAY_IMPORT_ORDER_POLICY_HPP

namespace dray
{
  struct ImportOrderPolicy
  {
    bool m_use_fixed_mesh_order;
    bool m_use_fixed_field_order;

    static ImportOrderPolicy general()
    {
      return { false, false };
    }

    static ImportOrderPolicy fixed_mesh_order()
    {
      return { true, false };
    }

    static ImportOrderPolicy fixed_field_order()
    {
      return { false, true };
    }

    static ImportOrderPolicy fixed()
    {
      return { true, true };
    }
  };
}

#endif//DRAY_IMPORT_ORDER_POLICY_HPP
