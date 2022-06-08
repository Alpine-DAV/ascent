// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ARRAY_REGISTRY_HPP
#define DRAY_ARRAY_REGISTRY_HPP

#include <list>
#include <stddef.h>

namespace dray
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

  // memory allocators
  static int device_allocator_id();
  // set a device allocator from outside ascent
  static bool device_allocator_id(int id);
  
  static int host_allocator_id();

  private:
  static std::list<ArrayInternalsBase *> m_arrays;

  static int m_device_allocator_id;
  static int m_host_allocator_id;
  static bool m_external_device_allocator;
};

} // namespace dray
#endif
