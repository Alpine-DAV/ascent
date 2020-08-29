// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef ASCENT_ARRAY_REGISTRY_HPP
#define ASCENT_ARRAY_REGISTRY_HPP

#include <list>
#include <stddef.h>

namespace ascent
{

namespace runtime
{

class ArrayInternalsBase;

class ArrayRegistry
{
  public:
  static void add_array (ArrayInternalsBase *array);
  static void remove_array (ArrayInternalsBase *array);
  static void release_device_res ();
  static size_t device_usage ();
  static size_t host_usage ();
  static int num_arrays ();

  private:
  static std::list<ArrayInternalsBase *> m_arrays;
};

} // namespace runtime
} // namespace ascent
#endif
