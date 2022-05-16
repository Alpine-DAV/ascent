// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Note: this file was derived from:
// https://gitlab.kitware.com/vtk/vtk-m/blob/master/vtkm/Matrix.h

#ifndef DRAY_MATRIX_HPP
#define DRAY_MATRIX_HPP

#include <dray/exports.hpp>
#include <dray/math.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

#include <assert.h>

namespace dray
{


template <typename T, int32 NumRow, int32 NumCol> class Matrix
{
  public:
  using ComponentType = T;
  static constexpr int32 NUM_ROWS = NumRow;
  static constexpr int32 NUM_COLUMNS = NumCol;

  template <typename TT, int32 NR, int32 NC>
  friend std::ostream &operator<< (std::ostream &os, const Matrix<TT, NR, NC> &matrix);

  /// Brackets are used to reference a matrix like a 2D array (i.e.
  /// matrix[row][column]).
  ///
  DRAY_EXEC
  const Vec<ComponentType, NUM_COLUMNS> &operator[] (int32 row_index) const
  {
    assert (row_index >= 0);
    assert (row_index < NUM_ROWS);
    return m_components[row_index];
  }

  /// Brackets are used to referens a matrix like a 2D array i.e.
  /// matrix[row][column].
  ///
  DRAY_EXEC
  Vec<ComponentType, NUM_COLUMNS> &operator[] (int32 row_index)
  {
    assert (row_index >= 0);
    assert (row_index < NUM_ROWS);
    return this->m_components[row_index];
  }

  /// Parentheses are used to reference a matrix using mathematical tuple
  /// notation i.e. matrix(row,column).
  ///
  DRAY_EXEC
  const ComponentType &operator() (int32 row_index, int32 col_index) const
  {
    assert (row_index >= 0);
    assert (row_index < NUM_ROWS);
    assert (col_index >= 0);
    assert (col_index < NUM_COLUMNS);
    return this->m_components[row_index][col_index];
  }

  /// Parentheses are used to reference a matrix using mathematical tuple
  /// notation i.e. matrix(row,column).
  ///
  DRAY_EXEC
  ComponentType &operator() (int32 row_index, int32 col_index)
  {
    assert (row_index >= 0);
    assert (row_index < NUM_ROWS);
    assert (col_index >= 0);
    assert (col_index < NUM_COLUMNS);
    return (*this)[row_index][col_index];
  }

  DRAY_EXEC
  void identity ()
  {
    for (int r = 0; r < NumRow; ++r)
      for (int c = 0; c < NumCol; ++c)
      {
        T val = T (0.f);
        if (c == r) val = T (1.f);
        m_components[r][c] = val;
      }
  }

  DRAY_EXEC
  Vec<T, NumCol> get_row (int32 row_idx) const
  {
    assert (row_idx >= 0 && row_idx < NumRow);
    Vec<T, NumCol> row;
    for (int32 i = 0; i < NumCol; i++)
    {
      row[i] = m_components[row_idx][i];
    }
    return row;
  }

  DRAY_EXEC
  void set_row (int32 row_idx, const Vec<T, NumCol> &row)
  {
    assert (row_idx >= 0 && row_idx < NumRow);
    m_components[row_idx] = row;
  }

  DRAY_EXEC
  Vec<T, NumRow> get_col (int32 col_idx) const
  {
    assert (col_idx >= 0 && col_idx < NumCol);
    Vec<T, NumRow> col;
    for (int32 ii = 0; ii < NumRow; ii++)
    {
      col[ii] = m_components[ii][col_idx];
    }
    return col;
  }

  DRAY_EXEC
  void set_col (const int32 col_idx, Vec<T, NumRow> col)
  {
    assert (col_idx >= 0 && col_idx < NumCol);
    for (int32 i = 0; i < NumRow; i++)
    {
      m_components[i][col_idx] = col[i];
    }
  }

  DRAY_EXEC
  Matrix<T, NumCol, NumRow> transpose () const
  {
    Matrix<T, NumCol, NumRow> result;
    for (int32 index = 0; index < NumRow; index++)
    {
      result.set_col (index, this->get_row (index));
    }
    return result;
  }

  template <int32 NumColRight>
  DRAY_EXEC Matrix<T, NumRow, NumColRight>
  operator* (const Matrix<T, NumCol, NumColRight> &right_factor) const
  {
    Matrix<T, NumRow, NumColRight> result;
    for (int32 row_index = 0; row_index < NumRow; row_index++)
    {
      for (int32 col_index = 0; col_index < NumColRight; col_index++)
      {
        T sum = T ((*this) (row_index, 0) * right_factor (0, col_index));
        for (int32 interal_index = 1; interal_index < NumCol; interal_index++)
        {
          sum = T (sum + ((*this) (row_index, interal_index) *
                          right_factor (interal_index, col_index)));
        }
        result (row_index, col_index) = sum;
      }
    }
    return result;
  }

  /// Standard matrix-vector multiplication.
  ///
  DRAY_EXEC Vec<T, NumRow> operator* (const Vec<T, NumCol> &right) const
  {
    Vec<T, NumRow> product;
    for (int32 row_index = 0; row_index < NumRow; row_index++)
    {
      product[row_index] = dot (get_row (row_index), right);
    }
    return product;
  }

  DRAY_EXEC void operator+= (const Matrix &other)
  {
    for (int32 row_idx = 0; row_idx < NumRow; row_idx++)
    {
      m_components[row_idx] += other.m_components[row_idx];
    }
  }

  /// Set all components to a single value.
  DRAY_EXEC void operator= (const T &single_val)
  {
    for (int32 row_idx = 0; row_idx < NumRow; row_idx++)
    {
      for (int32 col_idx = 0; col_idx < NumCol; col_idx++)
      {
        (*this) (row_idx, col_idx) = single_val;
      }
    }
  }

  /// Compute the product col * row.
  DRAY_EXEC
  static Matrix outer_product (const Vec<T, NumRow> &col, const Vec<T, NumCol> &row)
  {
    Matrix prod;
    for (int32 row_idx = 0; row_idx < NumRow; row_idx++)
    {
      for (int32 col_idx = 0; col_idx < NumCol; col_idx++)
      {
        prod (row_idx, col_idx) = col[row_idx] * row[col_idx];
      }
    }
    return prod;
  }

  private:
  Vec<Vec<ComponentType, NUM_COLUMNS>, NUM_ROWS> m_components;
};


namespace detail
{
// Used with MatrixLUPFactor.
template <typename T, int32 Size>
DRAY_EXEC void MatrixLUPFactorFindPivot (Matrix<T, Size, Size> &A,
                                         Vec<int32, Size> &permutation,
                                         int32 topCornerIndex,
                                         T &inversionParity,
                                         bool &valid)
{
  int32 maxRowIndex = topCornerIndex;
  T maxValue = fabs (A (maxRowIndex, topCornerIndex));
  for (int32 rowIndex = topCornerIndex + 1; rowIndex < Size; rowIndex++)
  {
    T compareValue = fabs (A (rowIndex, topCornerIndex));
    if (maxValue < compareValue)
    {
      maxValue = compareValue;
      maxRowIndex = rowIndex;
    }
  }

  if (maxValue < epsilon<T> ())
  {
    valid = false;
  }

  if (maxRowIndex != topCornerIndex)
  {
    // Swap rows in matrix.
    Vec<T, Size> maxRow = A.get_row (maxRowIndex);
    A.set_row (maxRowIndex, A.get_row (topCornerIndex));
    A.set_row (topCornerIndex, maxRow);

    // Record change in permutation matrix.
    int32 maxOriginalRowIndex = permutation[maxRowIndex];
    permutation[maxRowIndex] = permutation[topCornerIndex];
    permutation[topCornerIndex] = maxOriginalRowIndex;

    // Keep track of inversion parity.
    inversionParity = -inversionParity;
  }
}

// Used with MatrixLUPFactor
template <typename T, int32 Size>
DRAY_EXEC void
MatrixLUPFactorFindUpperTriangleElements (Matrix<T, Size, Size> &A, int32 topCornerIndex)
{
  // Compute values for upper triangle on row topCornerIndex
  for (int32 colIndex = topCornerIndex + 1; colIndex < Size; colIndex++)
  {
    A (topCornerIndex, colIndex) /= A (topCornerIndex, topCornerIndex);
  }

  // Update the rest of the matrix for calculations on subsequent rows
  for (int32 rowIndex = topCornerIndex + 1; rowIndex < Size; rowIndex++)
  {
    for (int32 colIndex = topCornerIndex + 1; colIndex < Size; colIndex++)
    {
      A (rowIndex, colIndex) -= A (rowIndex, topCornerIndex) * A (topCornerIndex, colIndex);
    }
  }
}
/// Performs an LUP-factorization on the given matrix using Crout's method. The
/// LU-factorization takes a matrix A and decomposes it into a lower triangular
/// matrix L and upper triangular matrix U such that A = LU. The
/// LUP-factorization also allows permutation of A, which makes the
/// decomposition always possible so long as A is not singular. In addition to
/// matrices L and U, LUP also finds permutation matrix P containing all zeros
/// except one 1 per row and column such that PA = LU.
///
/// The result is done in place such that the lower triangular matrix, L, is
/// stored in the lower-left triangle of A including the diagonal. The upper
/// triangular matrix, U, is stored in the upper-right triangle of L not
/// including the diagonal. The diagonal of U in Crout's method is all 1's (and
/// therefore not explicitly stored).
///
/// The permutation matrix P is represented by the permutation vector. If
/// permutation[i] = j then row j in the original matrix A has been moved to
/// row i in the resulting matrices. The permutation matrix P can be
/// represented by a matrix with p_i,j = 1 if permutation[i] = j and 0
/// otherwise. If using LUP-factorization to compute a determinant, you also
/// need to know the parity (whether there is an odd or even amount) of
/// inversions. An inversion is an instance of a smaller number appearing after
/// a larger number in permutation. Although you can compute the inversion
/// parity after the fact, this function keeps track of it with much less
/// compute resources. The parameter inversionParity is set to 1.0 for even
/// parity and -1.0 for odd parity.
///
/// Not all matrices (specifically singular matrices) have an
/// LUP-factorization. If the LUP-factorization succeeds, valid is set to true.
/// Otherwise, valid is set to false and the result is indeterminant.
///
template <typename T, int32 Size>
DRAY_EXEC void MatrixLUPFactor (Matrix<T, Size, Size> &A,
                                Vec<int32, Size> &permutation,
                                T &inversionParity,
                                bool &valid)
{
  // Initialize permutation.
  for (int32 index = 0; index < Size; index++)
  {
    permutation[index] = index;
  }
  inversionParity = T (1);
  valid = true;

  for (int32 rowIndex = 0; rowIndex < Size; rowIndex++)
  {
    MatrixLUPFactorFindPivot (A, permutation, rowIndex, inversionParity, valid);
    MatrixLUPFactorFindUpperTriangleElements (A, rowIndex);
  }
}
/// Use a previous factorization done with MatrixLUPFactor to solve the
/// system Ax = b.  Instead of A, this method takes in the LU and P
/// matrices calculated by MatrixLUPFactor from A. The x matrix is returned.
///
template <typename T, int32 Size>
DRAY_EXEC Vec<T, Size> MatrixLUPSolve (const Matrix<T, Size, Size> &LU,
                                       const Vec<int32, Size> &permutation,
                                       const Vec<T, Size> &b)
{
  // The LUP-factorization gives us PA = LU or equivalently A = inv(P)LU.
  // Substituting into Ax = b gives us inv(P)LUx = b or LUx = Pb.
  // Now consider the intermediate vector y = Ux.
  // Substituting in the previous two equations yields Ly = Pb.
  // Solving Ly = Pb is easy because L is triangular and P is just a
  // permutation.
  Vec<T, Size> y;
  for (int32 rowIndex = 0; rowIndex < Size; rowIndex++)
  {
    y[rowIndex] = b[permutation[rowIndex]];
    // Recall that L is stored in the lower triangle of LU including diagonal.
    for (int32 colIndex = 0; colIndex < rowIndex; colIndex++)
    {
      y[rowIndex] -= LU (rowIndex, colIndex) * y[colIndex];
    }
    y[rowIndex] /= LU (rowIndex, rowIndex);
  }

  // Now that we have y, we can easily solve Ux = y for x.
  Vec<T, Size> x;
  for (int32 rowIndex = Size - 1; rowIndex >= 0; rowIndex--)
  {
    // Recall that U is stored in the upper triangle of LU with the diagonal
    // implicitly all 1's.
    x[rowIndex] = y[rowIndex];
    for (int32 colIndex = rowIndex + 1; colIndex < Size; colIndex++)
    {
      x[rowIndex] -= LU (rowIndex, colIndex) * x[colIndex];
    }
  }

  return x;
}

} // namespace detail


// An interface to the result of LU decomposition.
// Useful to do multiple solves using same matrix but different right-hand-sides.
template <typename T, int32 S> class MatrixInverse
{
  public:
  DRAY_EXEC MatrixInverse (const Matrix<T, S, S> &A, bool &valid)
  {
    T inversionParity; // Unused
    m_LU = A;
    detail::MatrixLUPFactor (m_LU, m_permutation, inversionParity, m_valid);
    valid = m_valid;
  }

  DRAY_EXEC bool is_valid () const
  {
    return m_valid;
  }

  DRAY_EXEC Vec<T, S> operator* (const Vec<T, S> &right) const
  {
    return detail::MatrixLUPSolve (m_LU, m_permutation, right);
  }

  DRAY_EXEC Matrix<T, S, S> as_matrix () const
  {
    // We will use the decomposition to solve AX = I for X where X is
    // clearly the inverse of A.  Our solve method only works for vectors,
    // so we solve for one column of invA at a time.
    Matrix<T, S, S> invA;
    Vec<T, S> ICol;
    for (int32 i = 0; i < S; ++i)
      ICol[i] = T (0);

    for (int32 colIndex = 0; colIndex < S; colIndex++)
    {
      ICol[colIndex] = 1;
      invA.set_col (colIndex, operator* (ICol));
      ICol[colIndex] = 0;
    }
    return invA;
  }

  protected:
  Matrix<T, S, S> m_LU;
  Vec<int32, S> m_permutation;
  bool m_valid;
};


/// Find and return the inverse of the given matrix. If the matrix is singular,
/// the inverse will not be correct and valid will be set to false.
///
template <typename T, int32 Size>
DRAY_EXEC Matrix<T, Size, Size> matrix_inverse (const Matrix<T, Size, Size> &in, bool &valid)
{
  // TODO replace the body of this function by return MatrixInverse<T,Size>(in,valid).as_matrix();

  // First, we will make an LUP-factorization to help us.
  Matrix<T, Size, Size> LU = in;
  Vec<int32, Size> permutation;
  T inversionParity; // Unused
  detail::MatrixLUPFactor (LU, permutation, inversionParity, valid);

  // We will use the decomposition to solve AX = I for X where X is
  // clearly the inverse of A.  Our solve method only works for vectors,
  // so we solve for one column of invA at a time.
  Matrix<T, Size, Size> invA;
  Vec<T, Size> ICol;
  for (int32 i = 0; i < Size; ++i)
    ICol[i] = T (0);

  for (int32 colIndex = 0; colIndex < Size; colIndex++)
  {
    ICol[colIndex] = 1;
    Vec<T, Size> invACol = detail::MatrixLUPSolve (LU, permutation, ICol);
    ICol[colIndex] = 0;
    invA.set_col (colIndex, invACol);
  }
  return invA;
}

// Find x, the pre-image of vector y, under A, by performing LUPFactor and
// LUPSolve. Currently only square matrices are supported.
template <typename T, int32 S>
DRAY_EXEC Vec<T, S>
matrix_mult_inv (const Matrix<T, S, S> &A, const Vec<T, S> y, bool &valid)
{
  // TODO replace the body of this function by return MatrixInverse<T,S>(A,valid) * y;

  // LUP-factorize.
  Matrix<T, S, S> LU = A;
  Vec<int32, S> permutation;
  T inversionParity; // Unused
  detail::MatrixLUPFactor (LU, permutation, inversionParity, valid);

  // Solve Ax = y for x, using the above factorization..
  Vec<T, S> x = detail::MatrixLUPSolve (LU, permutation, y);

  return x;
}


template <typename TT, int32 NR, int32 NC>
std::ostream &operator<< (std::ostream &os, const Matrix<TT, NR, NC> &matrix)
{
  os << "Matrix[" << NR << ", " << NC << "]:\n";
  for (int32 i = 0; i < NR; ++i)
  {
    os << "  " << matrix[i] << "\n";
  }
  return os;
}

} // namespace dray
#endif
