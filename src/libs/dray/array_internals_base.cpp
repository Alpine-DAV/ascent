// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/array_internals_base.hpp>
#include <dray/array_registry.hpp>

namespace dray
{

ArrayInternalsBase::ArrayInternalsBase ()
{
  ArrayRegistry::add_array (this);
}

ArrayInternalsBase::~ArrayInternalsBase ()
{
  ArrayRegistry::remove_array (this);
}

} // namespace dray
