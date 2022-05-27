// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/vec.hpp>
#include <type_traits>
using namespace dray;

TEST (dray_vec, dray_vec)
{
  Vec3f vec = make_vec3f (1, 1, 1);
  Vec3f vec1 = make_vec3f (2, 2, 2);


  ASSERT_EQ (vec * 2, vec1);
  ASSERT_EQ (dot (vec, vec1), 6.f);
}
