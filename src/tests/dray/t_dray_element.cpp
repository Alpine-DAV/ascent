// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <iostream>
#include <stdio.h>

#include <dray/utils/png_encoder.hpp>

#include <dray/Element/element.hpp>

/// const int c_width = 1024;
/// const int c_height = 1024;
/// const int num_samples = 500000;

TEST (dray_element, dray_include_element)
{
  using T = dray::Float;
  constexpr unsigned int ncomp = 1;
  using DofT = dray::Vec<dray::Float, ncomp>;
  constexpr auto Quad = dray::ElemType::Quad;
  constexpr auto GeneralOrder = dray::Order::General;

  // Fake element data.
  DofT fake_dofs[64];
  int offsets[64];
  dray::init_counting (offsets, 64);
  // Arnold cat map (https://en.wikipedia.org/wiki/Arnold%27s_cat_map)
  const unsigned int mask = (1u << 8) - 1u;
  unsigned int q = 7;
  unsigned int p = 2;
  fake_dofs[0] = q;
  for (int ii = 1; ii < 64; ii++)
  {
    unsigned int qn = (2 * q + p) & mask;
    unsigned int pn = (q + p) & mask;
    q = qn;
    p = pn;
    fake_dofs[ii] = q;
  }

  std::cout << "Fake dof data:\n";
  for (int ii = 0; ii < 64; ii++)
  {
    std::cout << "  " << fake_dofs[ii];
  }
  std::cout << "\n";

  // Variable order implementation.
  dray::Element<2u, ncomp, Quad, GeneralOrder> quad_pg_2d;
  dray::Element<3u, ncomp, Quad, GeneralOrder> quad_pg_3d;

  // Fixed order implementation.
  dray::Element<2u, ncomp, Quad, 0> quad_p0_2d;
  quad_p0_2d.construct (0, { offsets, fake_dofs });
  dray::Element<2u, ncomp, Quad, 1> quad_p1_2d;
  quad_p1_2d.construct (0, { offsets, fake_dofs });
  dray::Element<2u, ncomp, Quad, 2> quad_p2_2d;
  quad_p2_2d.construct (0, { offsets, fake_dofs });
  dray::Element<2u, ncomp, Quad, 3> quad_p3_2d;
  quad_p3_2d.construct (0, { offsets, fake_dofs });

  dray::Element<3u, ncomp, Quad, 0> quad_p0_3d;
  quad_p0_3d.construct (0, { offsets, fake_dofs });
  dray::Element<3u, ncomp, Quad, 1> quad_p1_3d;
  quad_p1_3d.construct (0, { offsets, fake_dofs });
  dray::Element<3u, ncomp, Quad, 2> quad_p2_3d;
  quad_p2_3d.construct (0, { offsets, fake_dofs });
  dray::Element<3u, ncomp, Quad, 3> quad_p3_3d;
  quad_p3_3d.construct (0, { offsets, fake_dofs });


  // Evaluate at a reference point and compare values.
  const dray::Vec<T, 2u> ref2d{ { 0.3, 0.4 } };
  const dray::Vec<T, 3u> ref3d{ { 0.3, 0.4, 0.9 } };

  dray::Vec<DofT, 2u> ud2;
  dray::Vec<DofT, 3u> ud3;

  // 2D
  quad_pg_2d.construct (0, { offsets, fake_dofs }, 0);
  EXPECT_FLOAT_EQ (quad_pg_2d.eval_d (ref2d, ud2)[0], quad_p0_2d.eval (ref2d)[0]);

  quad_pg_2d.construct (0, { offsets, fake_dofs }, 1);
  EXPECT_FLOAT_EQ (quad_pg_2d.eval_d (ref2d, ud2)[0], quad_p1_2d.eval (ref2d)[0]);

  quad_pg_2d.construct (0, { offsets, fake_dofs }, 2);
  EXPECT_FLOAT_EQ (quad_pg_2d.eval_d (ref2d, ud2)[0], quad_p2_2d.eval (ref2d)[0]);

  /// quad_pg_2d.construct(0, {offsets, fake_dofs}, 3);
  /// EXPECT_FLOAT_EQ(quad_pg_2d.eval_d(ref2d, ud2)[0], quad_p3_2d.eval(ref2d)[0]); //TODO

  // 3D
  quad_pg_3d.construct (0, { offsets, fake_dofs }, 0);
  EXPECT_FLOAT_EQ (quad_pg_3d.eval_d (ref3d, ud3)[0], quad_p0_3d.eval (ref3d)[0]);

  quad_pg_3d.construct (0, { offsets, fake_dofs }, 1);
  EXPECT_FLOAT_EQ (quad_pg_3d.eval_d (ref3d, ud3)[0], quad_p1_3d.eval (ref3d)[0]);

  /// quad_pg_3d.construct(0, {offsets, fake_dofs}, 2);
  /// EXPECT_FLOAT_EQ(quad_pg_3d.eval_d(ref3d, ud3)[0], quad_p2_3d.eval(ref3d)[0]); //TODO

  /// quad_pg_3d.construct(0, {offsets, fake_dofs}, 3);
  /// EXPECT_FLOAT_EQ(quad_pg_3d.eval_d(ref3d, ud3)[0], quad_p3_3d.eval(ref3d)[0]); //TODO
}
