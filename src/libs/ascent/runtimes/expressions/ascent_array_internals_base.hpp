//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_ARRAY_INTERNALS_BASE_HPP
#define ASCENT_ARRAY_INTERNALS_BASE_HPP

#include <stddef.h>

namespace ascent
{

namespace runtime
{

class ArrayInternalsBase
{
  public:
  ArrayInternalsBase ();
  virtual ~ArrayInternalsBase ();
  virtual void release_device_ptr () = 0;
  virtual size_t device_alloc_size () = 0;
  virtual size_t host_alloc_size () = 0;
};

} // namespace runtime
} // namespace ascent

#endif
