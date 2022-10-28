//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_ARRAY_REGISTRY_HPP
#define ASCENT_ARRAY_REGISTRY_HPP

#include <list>
#include <stddef.h>
#include "ascent_memory_manager.hpp"

namespace ascent
{

namespace runtime
{

//-----------------------------------------------------------------------------
class ArrayInternalsBase;

//-----------------------------------------------------------------------------
class ArrayRegistry
{
  public:
    static void   add_array(ArrayInternalsBase *array);
    static void   remove_array(ArrayInternalsBase *array);

    static int    num_arrays();
    static size_t device_usage();
    static size_t host_usage();
    static size_t high_water_mark();
    static void   reset_high_water_mark();
    static void   release_device_resources();

    // book keeping methods
    static void   add_host_bytes(size_t bytes);
    static void   remove_host_bytes(size_t bytes);
    static void   add_device_bytes(size_t bytes);
    static void   remove_device_bytes(size_t bytes);

  private:
    static std::list<ArrayInternalsBase *> m_arrays;
    static size_t m_high_water_mark;
    static size_t m_device_bytes;
    static size_t m_host_bytes;

};

} // namespace runtime
} // namespace ascent
#endif
