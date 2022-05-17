// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/matrix.hpp>

TEST (dray_array, dray_matrix_inverse)
{
  dray::Matrix<float, 3, 3> mat;
  mat[0][0] = 1;
  mat[1][0] = 2;
  mat[2][0] = 3;

  mat[0][1] = 0;
  mat[1][1] = 1;
  mat[2][1] = 4;

  mat[0][2] = 5;
  mat[1][2] = 6;
  mat[2][2] = 0;

  std::cout << mat;

  bool valid;
  dray::Matrix<float, 3, 3> inv = dray::matrix_inverse (mat, valid);
  std::cout << inv;
  float abs_error = 0.0001f;

  ASSERT_NEAR (inv[0][0], -24.f, abs_error);
  ASSERT_NEAR (inv[1][0], 18.f, abs_error);
  ASSERT_NEAR (inv[2][0], 5.f, abs_error);

  ASSERT_NEAR (inv[0][1], 20.f, abs_error);
  ASSERT_NEAR (inv[1][1], -15.f, abs_error);
  ASSERT_NEAR (inv[2][1], -4.f, abs_error);

  ASSERT_NEAR (inv[0][2], -5.f, abs_error);
  ASSERT_NEAR (inv[1][2], 4.f, abs_error);
  ASSERT_NEAR (inv[2][2], 1.f, abs_error);

  std::cout << mat * inv << std::endl;

  std::cout << std::endl;

  std::cout << "--- Test set_row and set_column---" << std::endl;
  dray::Vec<float, 3> vec_0 = { 0, 0, 0 };
  dray::Vec<float, 3> vec_123 = { 1, 2, 3 };

  dray::Matrix<float, 3, 3> row_123;
  row_123[0] = vec_0;
  row_123[1] = vec_0;
  row_123[2] = vec_0;
  row_123.set_row (0, vec_123);

  dray::Matrix<float, 3, 3> column_123;
  column_123[0] = vec_0;
  column_123[1] = vec_0;
  column_123[2] = vec_0;
  column_123.set_col (0, vec_123);

  std::cout << "Row set (1,2,3)" << std::endl;
  std::cout << row_123;
  std::cout << "Column set (1,2,3)" << std::endl;
  std::cout << column_123;

  std::cout << std::endl;

  // Matrix multiplication works.
  /// std::cout << "--- Test B*C ---" << std::endl;
  /// dray::Matrix<float,3,3> B;
  /// B[0] = {1, 6, 11};
  /// B[1] = {2, 5, 12};
  /// B[2] = {3, 4, 13};

  /// dray::Matrix<float,3,3> C;
  /// C[0] = {-1, 2, 3};
  /// C[1] = { 0, 3, 1};
  /// C[2] = { 0,-5,-1};

  /// dray::Matrix<float,3,3> R;   // B*C computed by hand.
  /// R[0] = {-1,-35,-2};
  /// R[1] = {-2,-41,-1};
  /// R[2] = {-3,-47, 0};

  /// std::cout << "B, C, R, B*C" << std::endl;
  /// std::cout << B << C << R << B*C << std::endl;


  std::cout << "--- Test A*inv(A) and inv(A)*A ---" << std::endl;
  dray::Matrix<float, 3, 3> A;
  A[0] = { 3, 4, 0 };
  A[1] = { 2, 1, 6 };
  A[2] = { 1, 0, 5 };

  dray::Matrix<float, 3, 3> Y = dray::matrix_inverse (A, valid);
  std::cout << "+ Inverse " << (valid ? "succeeded" : "failed") << std::endl;

  if (valid)
  {
    std::cout << "A, inv(A), A*inv(A), inv(A)*A" << std::endl;
    std::cout << A << Y << A * Y << Y * A;
  }
}
