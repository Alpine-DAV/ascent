// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ARRAY_INTERNALS_BASE_HPP
#define DRAY_ARRAY_INTERNALS_BASE_HPP

#include <stddef.h>

namespace dray
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

} // namespace dray

#endif
