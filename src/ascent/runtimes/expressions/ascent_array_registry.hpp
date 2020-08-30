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
  static size_t high_water_mark();
  static int num_arrays ();
  static void add_device_bytes(size_t bytes);
  static void remove_device_bytes(size_t bytes);
  static void add_host_bytes(size_t bytes);
  static void remove_host_bytes(size_t bytes);

  private:
  static std::list<ArrayInternalsBase *> m_arrays;
  static size_t m_high_water_mark;
  static size_t m_device_bytes;
  static size_t m_host_bytes;
};

} // namespace runtime
} // namespace ascent
#endif
