// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/dray.hpp>
#include <dray/test.hpp>

TEST (dray_test, dray_test)
{
  dray::dray tracer;
  tracer.about ();
  dray::Tester tester;
  tester.raja_loop ();
}
