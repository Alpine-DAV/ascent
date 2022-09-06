// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <stdio.h>

#include <dray/Element/element.hpp>
#include <dray/filters/surface_triangle.hpp>
#include <dray/utils/png_encoder.hpp>

const int c_width = 1024;
const int c_height = 1024;
const int num_samples = 500000;

//
// dray_triangle_single
//
TEST (dray_triangle, dray_triangle_single)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "triangle");
  remove_test_image (output_file);

  using Coord = dray::Vec<float, 2>;
  using Color = dray::Vec<float, 4>;
  dray::Array<Color> img_buffer;

  // Define a linear triangle.
  Coord linear_triangle[3] = {
    { 0.0f, 0.0f },
    { .75f, 0.0f },
    { 0.0f, .75f },
  };

  // Define a flat quadratic triangle.
  Coord quadratic_triangle[6] = {
    { 0.0f, 0.0f }, { 0.4f, 0.0f }, { 0.8f, 0.0f },
    { 0.0f, 0.4f }, { 0.4f, 0.4f }, { 0.0f, 0.8f },
  };

  // Define a curvy quadratic triangle.
  const float curvy = 1.0f;
  const float sqt3 = pow (3.0f, 0.5);
  Coord curvy_quadratic_triangle[6] = {
    { 0.0f, 0.0f },     { 0.5f, 0.0f },     { 1.0f, 0.0f },
    { .25f, sqt3 / 4 }, { .75f, sqt3 / 4 }, { 0.5f, sqt3 / 2 },
  };
  const Coord centroid = { 0.5f, 0.5f / sqt3 };
  curvy_quadratic_triangle[1] = curvy_quadratic_triangle[1] * (1 - curvy) + centroid * curvy;
  curvy_quadratic_triangle[3] = curvy_quadratic_triangle[3] * (1 - curvy) + centroid * curvy;
  curvy_quadratic_triangle[4] = curvy_quadratic_triangle[4] * (1 - curvy) + centroid * curvy;

  /// dray::Array<Coord> nodes_array(linear_triangle, 3);
  /// const int poly_order = 1;
  /// dray::Array<Coord> nodes_array(quadratic_triangle, 6);
  /// const int poly_order = 2;
  dray::Array<Coord> nodes_array (curvy_quadratic_triangle, 6);
  const int poly_order = 2;

  img_buffer = dray::SurfaceTriangle ().execute<float> (c_width, c_height, nodes_array,
                                                        poly_order, num_samples);

  // Save image.
  dray::PNGEncoder png_encoder;
  png_encoder.encode ((float *)img_buffer.get_host_ptr (), c_width, c_height);
  png_encoder.save (output_file + ".png");

  // Check against benchmark.
  EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
}


//
// dray_triangle_derivatives
//
// Sample throughout the triangle and compare computed derivative
// to finite different approximation.
//
TEST (dray_triangle, dray_triangle_derivatives)
{
  using T = dray::Float;
  constexpr unsigned int ncomp = 1;
  using DofT = dray::Vec<T, ncomp>;

  constexpr auto Tri = dray::ElemType::Tri;
  constexpr auto GeneralOrder = dray::Order::General;

  const int poly_order = 2;
  const int sample_length = (1u << 11);

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
  /// std::cout << "Fake dof data:\n";
  /// for (int ii = 0; ii < 64; ii++)
  /// {
  ///   std::cout << "  " << fake_dofs[ii];
  /// }
  /// std::cout << "\n";


  // Variable order implementation.
  dray::Element<2u, ncomp, Tri, GeneralOrder> triangle;
  triangle.construct (0, { offsets, fake_dofs }, poly_order);

  // Sample and compare with finite difference.
  double maxAbsDiff = 0.0f;
  for (int jj = 0; jj < sample_length; jj++)
  {
    for (int ii = 0; ii < (sample_length - jj); ii++)
    {
      dray::Vec<T, 2u> ref = { T (ii) / sample_length, T (jj) / sample_length };
      dray::Vec<T, 2u> ref_i = { T (ii + 1) / sample_length, T (jj) / sample_length };
      dray::Vec<T, 2u> ref_j = { T (ii) / sample_length, T (jj + 1) / sample_length };

      dray::Vec<DofT, 2u> deriv;
      DofT val = triangle.eval_d (ref, deriv);

      dray::Vec<T, 2u> fin_diff;
      fin_diff[0] = (triangle.eval (ref_i) - val)[0] * (sample_length);
      fin_diff[1] = (triangle.eval (ref_j) - val)[0] * (sample_length);

      fin_diff[0] -= deriv[0][0];
      fin_diff[1] -= deriv[1][0];

      maxAbsDiff = fmax (maxAbsDiff, fin_diff.Normlinf ());
    }
  }

  fprintf (stdout, "maxAbsDiff == %.8f\n", maxAbsDiff);
  EXPECT_LT (maxAbsDiff, 1e-2);
}


//
// dray_tetrahedron_single
//
TEST (dray_triangle, dray_tetrahedron_single)
{
  using T = dray::Float;
  constexpr unsigned int ncomp = 3;
  using DofT = dray::Vec<T, ncomp>;

  constexpr auto Tri = dray::ElemType::Tri;
  constexpr auto GeneralOrder = dray::Order::General;

  const int p = 5;

  const int num_dofs = (p + 1) * (p + 2) * (p + 3) / 6;
  dray::Vec<T, 3> identity_dofs[num_dofs];
  int offsets[num_dofs];
  dray::init_counting (offsets, num_dofs);
  int dof_idx = 0;

  for (int kk = 0; kk <= p; kk++)
  {
    for (int jj = 0; jj <= p - kk; jj++)
    {
      for (int ii = 0; ii <= p - kk - jj; ii++)
      {
        identity_dofs[dof_idx++] = { T (ii) / p, T (jj) / p, T (kk) / p };
      }
    }
  }

  dray::Element<3u, ncomp, Tri, GeneralOrder> my_tetrahedron;
  my_tetrahedron.construct (0, { offsets, identity_dofs }, p);

  for (int kk = 0; kk <= p; kk++)
  {
    for (int jj = 0; jj <= p - kk; jj++)
    {
      for (int ii = 0; ii <= p - kk - jj; ii++)
      {
        dray::Vec<T, 3> ref = { T (ii) / p, T (jj) / p, T (kk) / p };
        dray::Vec<T, 3> loc = my_tetrahedron.eval (ref);
        EXPECT_FLOAT_EQ (ref[0], loc[0]);
        EXPECT_FLOAT_EQ (ref[1], loc[1]);
        EXPECT_FLOAT_EQ (ref[2], loc[2]);
      }
    }
  }
}


//
// dray_tetrahedron_derivatives
//
TEST (dray_triangle, dray_tetrahedron_derivatives)
{
  using T = dray::Float;
  constexpr unsigned int ncomp = 1;
  using DofT = dray::Vec<T, ncomp>;

  constexpr auto Tri = dray::ElemType::Tri;
  constexpr auto GeneralOrder = dray::Order::General;

  const int poly_order = 2;
  const int sample_length = (1u << 8);

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
  dray::Element<3u, ncomp, Tri, GeneralOrder> tetrahedron;
  tetrahedron.construct (0, { offsets, fake_dofs }, poly_order);

  // Sample and compare with finite difference.
  double maxAbsDiff = 0.0f;
  for (int kk = 0; kk < sample_length; kk++)
  {
    for (int jj = 0; jj < (sample_length - kk); jj++)
    {
      for (int ii = 0; ii < (sample_length - kk - jj); ii++)
      {
        dray::Vec<T, 3u> ref = { T (ii) / sample_length, T (jj) / sample_length,
                                 T (kk) / sample_length };
        dray::Vec<T, 3u> ref_i = { T (ii + 1) / sample_length,
                                   T (jj) / sample_length, T (kk) / sample_length };
        dray::Vec<T, 3u> ref_j = { T (ii) / sample_length, T (jj + 1) / sample_length,
                                   T (kk) / sample_length };
        dray::Vec<T, 3u> ref_k = { T (ii) / sample_length, T (jj) / sample_length,
                                   T (kk + 1) / sample_length };

        dray::Vec<DofT, 3u> deriv;
        DofT val = tetrahedron.eval_d (ref, deriv);

        dray::Vec<T, 3u> fin_diff;
        fin_diff[0] = (tetrahedron.eval (ref_i) - val)[0] * (sample_length);
        fin_diff[1] = (tetrahedron.eval (ref_j) - val)[0] * (sample_length);
        fin_diff[2] = (tetrahedron.eval (ref_k) - val)[0] * (sample_length);

        fin_diff[0] -= deriv[0][0];
        fin_diff[1] -= deriv[1][0];
        fin_diff[2] -= deriv[2][0];

        maxAbsDiff = fmax (maxAbsDiff, fin_diff.Normlinf ());
      }
    }
  }

  fprintf (stdout, "maxAbsDiff == %.8f\n", maxAbsDiff);
  EXPECT_LT (maxAbsDiff, 0.5);
}
