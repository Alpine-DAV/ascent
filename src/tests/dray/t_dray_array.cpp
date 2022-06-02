// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/array.hpp>

TEST (dray_array, dray_array_basic)
{
  dray::Array<int> int_array;
  int_array.resize (2);
  int *host = int_array.get_host_ptr ();
  host[0] = 0;
  host[1] = 1;

  int *host2 = int_array.get_host_ptr ();
  ASSERT_EQ (host2[0], 0);
  ASSERT_EQ (host2[1], 1);
}
