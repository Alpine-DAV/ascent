// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/array.hpp>
#include <dray/array_registry.hpp>
#include <dray/dray.hpp>

TEST (dray_array, dray_registry_basic)
{
  dray::ArrayRegistry::summary();
  ASSERT_EQ (dray::ArrayRegistry::number_of_arrays(), 0);

  dray::Array<int> int_array;
  ASSERT_EQ (dray::ArrayRegistry::number_of_arrays(), 1);

  int_array.resize(2);
  int *host = int_array.get_host_ptr();
  host[0] = 0;
  host[1] = 1;

  size_t host_usage = dray::ArrayRegistry::host_usage();

  ASSERT_EQ(host_usage, 2 * sizeof (int));
  
  
  dray::Array<float> float_array;
  float_array.resize(10);
  float *host_ptr = float_array.get_host_ptr();
  ASSERT_EQ (dray::ArrayRegistry::number_of_arrays(), 2);

  host_usage = dray::ArrayRegistry::host_usage();
  ASSERT_EQ(host_usage, (2 * sizeof (int)) + (10 * sizeof(float)));


  std::cout << std::endl <<  "Summary after int and float array creation:" << std::endl;
  dray::ArrayRegistry::summary();

  size_t dev_usage = dray::ArrayRegistry::device_usage();

  // we should not have allocated anything yet
  ASSERT_EQ (dev_usage, 0);
  int *dev = int_array.get_device_ptr();

  std::cout << std::endl << "Summary post get_device_ptr:" << std::endl;
  dray::ArrayRegistry::summary();

  dev_usage = dray::ArrayRegistry::device_usage ();
  // not we should have two ints
  if (dray::dray::device_enabled ())
  {
    ASSERT_EQ(dev_usage, 2 * sizeof (int));
  }
  else
  {
    ASSERT_EQ(dev_usage, 0);
  }

  dray::ArrayRegistry::release_device_res ();
  dev_usage = dray::ArrayRegistry::device_usage ();
  ASSERT_EQ(dev_usage, 0);

  std::cout << std::endl << "Summary post release_device_res:" << std::endl;
  dray::ArrayRegistry::summary();

}
