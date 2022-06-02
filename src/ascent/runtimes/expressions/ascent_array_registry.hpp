//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
  static void reset_high_water_mark();

  // memory allocators
  static int device_allocator_id();
  // set a device allocator from outside ascent
  static void device_allocator_id(int id);

  static int host_allocator_id();
  // TODO: setting a new host allocator invalidates
  // all memory, and this could be set in the middle of
  // everything. While we have a path to deallocat and synch
  // device resources to the host side, we would
  // need a method to realloc all host arrays.
  //static void host_allocator_id(int);

  private:
  static std::list<ArrayInternalsBase *> m_arrays;
  static size_t m_high_water_mark;
  static size_t m_device_bytes;
  static size_t m_host_bytes;
  static int m_device_allocator_id;
  static int m_host_allocator_id;
  static bool m_external_device_allocator;
};

} // namespace runtime
} // namespace ascent
#endif
