// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BERSTEIN_BASIS_HPP
#define DRAY_BERSTEIN_BASIS_HPP

#include <dray/binomial.hpp> // For BinomRow, to help BernsteinBasis.
#include <dray/constants.hpp>
#include <dray/range.hpp>
#include <dray/vec.hpp>

#include <string.h> // memcpy()

namespace dray
{

template <int32 RefDim> struct BernsteinBasis
{
  // Internals
  int32 p;
  Float *m_aux_mem_ptr; // Don't use this.

  // Public
  /// DRAY_EXEC void init_shape(int32 _p, T *aux_mem_ptr) { p = _p; m_aux_mem_ptr = aux_mem_ptr; }
  DRAY_EXEC void init_shape (int32 _p)
  {
    p = _p;
  }

  static constexpr int32 ref_dim = RefDim;
  DRAY_EXEC int32 get_el_dofs () const
  {
    return pow (p + 1, RefDim);
  }

  /// DRAY_EXEC void set_aux_mem_ptr(T *aux_mem_ptr) { m_aux_mem_ptr = aux_mem_ptr; }

  // The number of auxiliary elements needed for member aux_mem.
  // For each reference dim, need a row for values and a row for derivatives.
  // Can compute tensor-product on the fly from these rows.
  static int32 get_aux_req (int32 p)
  {
    return 2 * RefDim * (p + 1);
  }
  DRAY_EXEC int32 get_aux_req () const
  {
    return 2 * RefDim * (p + 1);
  }
  DRAY_EXEC static bool is_aux_req ()
  {
    return false;
  }

  // Linear combination of value functions, and linear combinations of derivative functions.
  // This is to evaluate a transformmation using a given set of control points at a given reference points.
  template <typename CoeffIterType, int32 PhysDim, int32 IterDim = RefDim>
  static DRAY_EXEC void linear_combo (const Vec<Float, IterDim> &xyz,
                                      const CoeffIterType &coeff_iter,
                                      int32 p_order,
                                      Vec<Float, PhysDim> &result_val,
                                      Vec<Vec<Float, PhysDim>, IterDim> &result_deriv);

  template <typename CoeffIterType, int32 PhysDim, int32 IterDim = RefDim>
  DRAY_EXEC void linear_combo (const Vec<Float, IterDim> &xyz,
                               const CoeffIterType &coeff_iter,
                               Vec<Float, PhysDim> &result_val,
                               Vec<Vec<Float, PhysDim>, IterDim> &result_deriv)
  {
    linear_combo<CoeffIterType, PhysDim, IterDim> (xyz, coeff_iter, p, result_val, result_deriv);
  }


  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo_divmod (const Vec<Float, RefDim> &xyz,
                                      const CoeffIterType &coeff_iter,
                                      Vec<Float, PhysDim> &result_val,
                                      Vec<Vec<Float, PhysDim>, RefDim> &result_deriv);

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo_old (const Vec<Float, RefDim> &xyz,
                                   const CoeffIterType &coeff_iter,
                                   Vec<Float, PhysDim> &result_val,
                                   Vec<Vec<Float, PhysDim>, RefDim> &result_deriv);

  //DRAY_EXEC static bool is_inside (const Vec<Float, RefDim> ref_pt)
  //{
  //  for (int32 rdim = 0; rdim < RefDim; rdim++)
  //    if (!(0 <= ref_pt[rdim] && ref_pt[rdim] <= 1)) // TODO
  //      return false;
  //  return true;
  //}

  // If just want raw shape values/derivatives,
  // stored in memory, to do something with them later:
  ////DRAY_EXEC void calc_shape_dshape(const Vec<T,RefDim> &ref_pt, T *shape_val, Vec<T,RefDim> *shape_deriv) const;   //TODO

  // ref_box is a list of ranges defining the corners of a sub-element in reference space.
  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC static Vec<Float, PhysDim> get_sub_coefficient (const Range *ref_box,
                                                            const CoeffIterType &coeff_iter,
                                                            int32 p,
                                                            int32 i0,
                                                            int32 i1 = 0,
                                                            int32 i2 = 0);

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC Vec<Float, PhysDim> get_sub_coefficient (const Range *ref_box,
                                                     const CoeffIterType &coeff_iter,
                                                     int32 i0,
                                                     int32 i1 = 0,
                                                     int32 i2 = 0)
  {
    return get_sub_coefficient<CoeffIterType, PhysDim> (ref_box, p, i0, i1, i2);
  }

  DRAY_EXEC static void
  splitting_matrix_1d_seq (int32 p, int32 ii, Float t0, Float t1, Float *W);

  // Get the component of the sub-element splitting matrix:
  //     C_ii = B_jj * M_jj_ii
  //   Put in the original coefficients in the list B,
  //   get out the sub-element coefficients in the list C.
  //
  // M is not triangular for an arbitrary segment 0 < t0 <= t1 < 1.
  //
  // Assumes that 0 < t0 <= t1 < 1.
  DRAY_EXEC static Float
  splitting_matrix_1d_comp (int32 p, int32 ii, int32 jj, Float t0, Float t1);

  // Get all the components of the sub-element splitting matrix, all jj for a single ii:
  //     C_ii = B_jj * M_jj_ii
  //
  // M is triangular for strictly left or right segments.
  // This function only sets the nonzeros entries (W[0] .. W[ii] inclusive).
  //
  // Assumes that 0 == t0 <= t1 <= 1.
  DRAY_EXEC static void
  splitting_matrix_1d_left_seq (int32 p, int32 ii, Float t1, Float *W);

  // Same thing as splitting_matrix_1d_left_seq() but for right and in reverse order: W[jj] = W_{ii, p-jj}.
  // This function only sets the nonzeros entries (W[ii] .. W[0] inclusive).
  //
  // Assumes that 0 <= t0 <= t1 == 1.
  DRAY_EXEC static void
  splitting_matrix_1d_right_seq (int32 p, int32 ii, Float t0, Float *W);


  // Copies the element data into a MultiVec struct, then does double-DeCasteljau splitting
  // in-place along each axis. The result is the set of coefficients for the subdivided element.
  template <typename CoeffIterType, uint32 PhysDim, uint32 p_order>
  DRAY_EXEC static MultiVec<Float, 3, PhysDim, p_order>
  decasteljau_3d (const Range *ref_box, const CoeffIterType &coeff_iter); // TODO change the name

}; // BernsteinBasis


struct DeCasteljau
{
  // Finds the left edge of the DeCasteljau triangle. This is a 1D operator.
  // However, the coefficients can be multidimensional arrays (MultiVec).
  template <typename MultiArrayT, int32 POrder = -1>
  DRAY_EXEC static void
  split_inplace_left (MultiArrayT &elem_data, Float t1, uint32 p_order = 0)
  {
    const uint32 p = (POrder >= 0 ? POrder : p_order); // p is Maybe a template parameter.

    for (int32 ii = 1; ii <= p; ii++)
    {
      // TODO do the below stuff component-wise, eliminate multi-dimensional tmp.
      auto tmp = elem_data[ii - 1]; // multi-dimensional buffer.
      for (int32 jj = ii; jj <= p_order; jj++)
      {
        tmp = tmp * (1 - t1) + elem_data[jj] * t1;
        elem_data[jj].swap (tmp);
      }
    }
  }

  // Finds the right edge of the DeCasteljau triangle. This is a 1D operator.
  // However, the coefficients can be multidimensional arrays (MultiVec).
  template <typename MultiArrayT, int32 POrder = -1>
  DRAY_EXEC static void
  split_inplace_right (MultiArrayT &elem_data, Float t0, uint32 p_order = 0)
  {
    const uint32 p = (POrder >= 0 ? POrder : p_order); // p is Maybe a template parameter.

    for (int32 ii = p - 1; ii >= 0; ii--)
    {
      // TODO do the below stuff component-wise, eliminate multi-dimensional tmp.
      auto tmp = elem_data[ii + 1]; // multi-dimensional buffer.
      for (int32 jj = ii; jj >= 0; jj--)
      {
        tmp = tmp * t0 + elem_data[jj] * (1 - t0);
        elem_data[jj].swap (tmp);
      }
    }
  }
};


namespace detail_BernsteinBasis
{
// Helper functions to access the auxiliary memory space.
DRAY_EXEC static int32 aux_mem_val_offset (int32 p, int32 rdim)
{
  return (2 * rdim) * (p + 1);
}
DRAY_EXEC static int32 aux_mem_deriv_offset (int32 p, int32 rdim)
{
  return (2 * rdim + 1) * (p + 1);
}

template <typename T> DRAY_EXEC T dpow (const T &base, const int32 &exp)
{
  T res = 1.f;
  for (int i = 0; i < exp; ++i)
  {
    res *= base;
  }
  return res;
}
// Bernstein evaluator using pow(). Assumes binomial coefficient is the right one (p choose k).
// template <typename T>
// DRAY_EXEC
// static void calc_shape_dshape_1d_single(const int32 p, const int32 k, const T x, const int32 bcoeff, T &u, T &d)
//{
//  if (p == 0)
//  {
//    u = 1.;
//    d = 0.;
//  }
//  else
//  {
//    u = bcoeff * pow(x, k) * pow(1-x, p-k);
//    d = (k == 0 ? -p * pow(1-x, p-1) :
//         k == p ?  p * pow(x, p-1)    :
//                  (k - p*x) * pow(x, k-1) * pow(1-x, p-k-1)) * bcoeff;
//  }
//}
template <typename T>
DRAY_EXEC static void
calc_shape_dshape_1d_single (const int32 p, const int32 k, const T x, const int32 bcoeff, T &u, T &d)
{
  if (p == 0)
  {
    u = 1.;
    d = 0.;
  }
  else
  {
    u = bcoeff * dpow (x, k) * dpow (1 - x, p - k);
    d = (k == 0 ? -p * dpow (1 - x, p - 1) :
                  k == p ? p * dpow (x, p - 1) :
                           (k - p * x) * dpow (x, k - 1) * dpow (1 - x, p - k - 1)) *
        bcoeff;
  }
}

// Bernstein evaluator adapted from MFEM.
template <typename T>
DRAY_EXEC static void
calc_shape_dshape_1d (const int32 p, const T x, const T y, T *u, T *d)
{
  if (p == 0)
  {
    u[0] = 1.;
    d[0] = 0.;
  }
  else
  {
    // Assume that binomial coefficients are already sitting in the arrays u[], d[].
    const double xpy = x + y, ptx = p * x;
    double z = 1.;

    int i;
    for (i = 1; i < p; i++)
    {
      // d[i] = b[i]*z*(i*xpy - ptx);
      d[i] = d[i] * z * (i * xpy - ptx);
      z *= x;
      // u[i] = b[i]*z;
      u[i] = u[i] * z;
    }
    d[p] = p * z;
    u[p] = z * x;
    z = 1.;
    for (i--; i > 0; i--)
    {
      d[i] *= z;
      z *= y;
      u[i] *= z;
    }
    d[0] = -p * z;
    u[0] = z * y;
  }
}

template <typename T>
DRAY_EXEC static void calc_shape_1d (const int32 p, const T x, const T y, T *u)
{
  if (p == 0)
  {
    u[0] = 1.;
  }
  else
  {
    // Assume that binomial coefficients are already sitting in the array u[].
    double z = 1.;
    int i;
    for (i = 1; i < p; i++)
    {
      z *= x;
      u[i] = u[i] * z;
    }
    u[p] = z * x;
    z = 1.;
    for (i--; i > 0; i--)
    {
      z *= y;
      u[i] *= z;
    }
    u[0] = z * y;
  }
}

template <typename T>
DRAY_EXEC static void calc_dshape_1d (const int32 p, const T x, const T y, T *d)
{
  if (p == 0)
  {
    d[0] = 0.;
  }
  else
  {
    // Assume that binomial coefficients are already sitting in the array d[].
    const double xpy = x + y, ptx = p * x;
    double z = 1.;

    int i;
    for (i = 1; i < p; i++)
    {
      d[i] = d[i] * z * (i * xpy - ptx);
      z *= x;
    }
    d[p] = p * z;
    z = 1.;
    for (i--; i > 0; i--)
    {
      d[i] *= z;
      z *= y;
    }
    d[0] = -p * z;
  }
}

} // namespace detail_BernsteinBasis


// TODO use recursive templates and clean this file up.

template <int32 RefDim>
template <typename CoeffIterType, int32 PhysDim, int32 IterDim>
DRAY_EXEC void
BernsteinBasis<RefDim>::linear_combo (const Vec<Float, IterDim> &xyz,
                                      const CoeffIterType &coeff_iter,
                                      int32 p,
                                      Vec<Float, PhysDim> &result_val,
                                      Vec<Vec<Float, PhysDim>, IterDim> &result_deriv)
{
  // No pow(), no aux_mem_ptr.
  //
  // Directly evaluate a Bernstein polynomial with a hybrid of Horner's rule and accumulation of powers:
  //     V = 0.0;  xpow = 1.0;
  //     for(i)
  //     {
  //       V = V*(1-x) + C[i]*xpow*nchoosek(p,i);
  //       xpow *= x;
  //     }
  //
  // Indirectly evaluate a high-order Bernstein polynomial, by directly evaluating
  // the two parent lower-order Bernstein polynomials, and mixing with weights {(1-x), x}.
  //
  // Indirectly evaluate the derivative of a high-order Bernstein polynomial, by directly
  // evaluating the two parent lower-order Bernstein polynomials, and mixing with weights {-p, p}.

  Vec<Float, PhysDim> zero;
  zero = 0;

  const Float x = (IterDim > 0 ? xyz[0] : 0.0);
  const Float y = (IterDim > 1 ? xyz[1] : 0.0);
  const Float z = (IterDim > 2 ? xyz[2] : 0.0);
  const Float xbar = 1.0 - x;
  const Float ybar = 1.0 - y;
  const Float zbar = 1.0 - z;

  const int32 p1 = (IterDim >= 1 ? p : 0);
  const int32 p2 = (IterDim >= 2 ? p : 0);
  const int32 p3 = (IterDim >= 3 ? p : 0);

  int32 B[MaxPolyOrder];
  {
    BinomRowIterator binom_coeff_generator;
    binom_coeff_generator.construct (p - 1);
    for (int32 ii = 0; ii <= p - 1; ii++)
    {
      B[ii] = *binom_coeff_generator;
      binom_coeff_generator.next ();
    }
  }

  int32 cidx = 0; // Index into element dof indexing space.

  // Compute and combine order (p-1) values to get order (p) values/derivatives.
  // https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties

  Vec<Float, PhysDim> val_x, val_y, val_z;
  Vec<Float, PhysDim> deriv_x;
  Vec<Vec<Float, PhysDim>, 2> deriv_xy;
  Vec<Vec<Float, PhysDim>, 3> deriv_xyz;

  // Level3 set up.
  Float zpow = 1.0;
  Vec<Vec<Float, PhysDim>, 3> val_z_L,
  val_z_R; // Second/third columns are derivatives in lower level.
  val_z_L = zero;
  val_z_R = zero;
  for (int32 ii = 0; ii <= p3; ii++)
  {
    // Level2 set up.
    Float ypow = 1.0;
    Vec<Vec<Float, PhysDim>, 2> val_y_L, val_y_R; // Second column is derivative in lower level.
    val_y_L = zero;
    val_y_R = zero;

    for (int32 jj = 0; jj <= p2; jj++)
    {
      // Level1 set up.
      Float xpow = 1.0;
      Vec<Float, PhysDim> val_x_L = zero, val_x_R = zero; // L and R can be combined --> val, deriv.
      Vec<Float, PhysDim> C = coeff_iter[cidx++];
      for (int32 kk = 1; kk <= p1; kk++)
      {
        // Level1 accumulation.
        val_x_L = val_x_L * xbar + C * (B[kk - 1] * xpow);
        C = coeff_iter[cidx++];
        val_x_R = val_x_R * xbar + C * (B[kk - 1] * xpow);
        xpow *= x;
      } // kk

      // Level1 result.
      val_x = (p1 > 0 ? val_x_L * xbar + val_x_R * x : C);
      deriv_x = (val_x_R - val_x_L) * p1;

      // Level2 accumulation.
      if (jj > 0)
      {
        val_y_R[0] = val_y_R[0] * ybar + val_x * (B[jj - 1] * ypow);
        val_y_R[1] = val_y_R[1] * ybar + deriv_x * (B[jj - 1] * ypow);
        ypow *= y;
      }
      if (jj < p2)
      {
        val_y_L[0] = val_y_L[0] * ybar + val_x * (B[jj] * ypow);
        val_y_L[1] = val_y_L[1] * ybar + deriv_x * (B[jj] * ypow);
      }
    } // jj

    // Level2 result.
    val_y = (p2 > 0 ? val_y_L[0] * ybar + val_y_R[0] * y : val_x);
    deriv_xy[0] = (p2 > 0 ? val_y_L[1] * ybar + val_y_R[1] * y : deriv_x);
    deriv_xy[1] = (val_y_R[0] - val_y_L[0]) * p2;

    // Level3 accumulation.
    if (ii > 0)
    {
      val_z_R[0] = val_z_R[0] * zbar + val_y * (B[ii - 1] * zpow);
      val_z_R[1] = val_z_R[1] * zbar + deriv_xy[0] * (B[ii - 1] * zpow);
      val_z_R[2] = val_z_R[2] * zbar + deriv_xy[1] * (B[ii - 1] * zpow);
      zpow *= z;
    }
    if (ii < p3)
    {
      val_z_L[0] = val_z_L[0] * zbar + val_y * (B[ii] * zpow);
      val_z_L[1] = val_z_L[1] * zbar + deriv_xy[0] * (B[ii] * zpow);
      val_z_L[2] = val_z_L[2] * zbar + deriv_xy[1] * (B[ii] * zpow);
    }
  } // ii

  // Level3 result.
  val_z = (p3 > 0 ? val_z_L[0] * zbar + val_z_R[0] * z : val_y);
  deriv_xyz[0] = (p3 > 0 ? val_z_L[1] * zbar + val_z_R[1] * z : deriv_xy[0]);
  deriv_xyz[1] = (p3 > 0 ? val_z_L[2] * zbar + val_z_R[2] * z : deriv_xy[1]);
  deriv_xyz[2] = (val_z_R[0] - val_z_L[0]) * p3;

  result_val = val_z;
  if (IterDim > 0) result_deriv[0] = deriv_xyz[0];
  if (IterDim > 1) result_deriv[1] = deriv_xyz[1];
  if (IterDim > 2) result_deriv[2] = deriv_xyz[2];
}


template <int32 RefDim>
template <typename CoeffIterType, int32 PhysDim>
DRAY_EXEC void
BernsteinBasis<RefDim>::linear_combo_divmod (const Vec<Float, RefDim> &xyz,
                                             const CoeffIterType &coeff_iter,
                                             Vec<Float, PhysDim> &result_val,
                                             Vec<Vec<Float, PhysDim>, RefDim> &result_deriv)
{
  // The simple way, using pow() and not aux_mem_ptr.

  // Initialize output parameters.
  result_val = 0;
  for (int32 rdim = 0; rdim < RefDim; rdim++)
    result_deriv[rdim] = 0;

  const int32 pp1 = p + 1;

  //
  // Accumulate the tensor product components.
  // Read each control point once.
  //

  // Set up index formulas.
  // First coordinate is outermost, encompasses all. Last is innermost, encompases (p+1).
  int32 stride[RefDim];
  stride[0] = 1;
  for (int32 rdim = 1; rdim < RefDim; rdim++)
    stride[rdim] = stride[rdim - 1] * (pp1);
  int32 el_dofs = stride[RefDim - 1] * (pp1);

  int32 ii[RefDim];
  Float shape_val[RefDim];
  Float shape_deriv[RefDim];

  // Construct the binomial coefficients (part of the shape functions).
  BinomRowIterator binom_coeff[RefDim];
  for (int32 rdim = 0; rdim < RefDim; rdim++)
  {
    ii[rdim] = 0;
    binom_coeff[rdim].construct (p);
    detail_BernsteinBasis::calc_shape_dshape_1d_single (p, 0, xyz[rdim],
                                                        *binom_coeff[rdim],
                                                        shape_val[rdim],
                                                        shape_deriv[rdim]);
  }

  // Iterate over degrees of freedom, i.e., iterate over control point values.
  for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
  {
    // Compute index and new shape values.
    for (int32 rdim_in = 0; rdim_in < RefDim; rdim_in++)
    {
      int32 tmp_ii = (dof_idx / stride[rdim_in]) % (pp1);
      if (tmp_ii != ii[rdim_in]) // On the edge of the next step in ii[rdim_in].
      {
        ii[rdim_in] = tmp_ii;
        binom_coeff[rdim_in].next ();
        detail_BernsteinBasis::calc_shape_dshape_1d_single (p, ii[rdim_in], xyz[rdim_in],
                                                            *binom_coeff[rdim_in],
                                                            shape_val[rdim_in],
                                                            shape_deriv[rdim_in]);
      }
    }

    // Compute tensor product shape.
    Float t_shape_val = shape_val[0];
    for (int32 rdim_in = 1; rdim_in < RefDim; rdim_in++)
      t_shape_val *= shape_val[rdim_in];

    // Multiply control point value, accumulate value.
    const Vec<Float, PhysDim> ctrl_val = coeff_iter[dof_idx];
    result_val += ctrl_val * t_shape_val;

    for (int32 rdim_out = 0; rdim_out < RefDim; rdim_out++) // Over the derivatives.
    {
      Float t_shape_deriv = shape_deriv[rdim_out];
      int32 rdim_in;
      for (rdim_in = 0; rdim_in < rdim_out; rdim_in++) // Over the tensor dimensions.
        t_shape_deriv *= shape_val[rdim_in];
      for (++rdim_in; rdim_in < RefDim; rdim_in++) // Over the tensor dimensions.
        t_shape_deriv *= shape_val[rdim_in];

      // Multiply control point value, accumulate value.
      result_deriv[rdim_out] += ctrl_val * t_shape_deriv;
    }
  }
}


template <int32 RefDim>
template <typename CoeffIterType, int32 PhysDim>
DRAY_EXEC void
BernsteinBasis<RefDim>::linear_combo_old (const Vec<Float, RefDim> &xyz,
                                          const CoeffIterType &coeff_iter,
                                          Vec<Float, PhysDim> &result_val,
                                          Vec<Vec<Float, PhysDim>, RefDim> &result_deriv)
{
  // Initialize output parameters.
  result_val = 0;
  for (int32 rdim = 0; rdim < RefDim; rdim++)
    result_deriv[rdim] = 0;

  const int32 pp1 = p + 1;

  // Make names for the rows of auxiliary memory.
  Float *val_i[RefDim];
  Float *deriv_i[RefDim];
  for (int32 rdim = 0; rdim < RefDim; rdim++)
  {
    val_i[rdim] = m_aux_mem_ptr + detail_BernsteinBasis::aux_mem_val_offset (p, rdim);
    deriv_i[rdim] = m_aux_mem_ptr + detail_BernsteinBasis::aux_mem_deriv_offset (p, rdim);
  }

  // The first two rows will be used specially.
  Float *&val_0 = val_i[0];
  Float *&deriv_0 = deriv_i[0];

  //
  // Populate shape values and derivatives.
  //

  // Fill the first two rows with binomial coefficients.
  BinomRow<Float>::fill_single_row (p, val_0);
  memcpy (deriv_0, val_0, pp1 * sizeof (Float));

  // Compute shape values and derivatives for latter dimensions.
  for (int32 rdim = 1; rdim < RefDim; rdim++)
  {
    // Copy binomial coefficients.
    memcpy (val_i[rdim], val_0, pp1 * sizeof (Float));
    memcpy (deriv_i[rdim], val_0, pp1 * sizeof (Float));

    // Compute shape values and derivatives.
    const Float x_i = xyz[rdim];
    detail_BernsteinBasis::calc_shape_1d<Float> (p, x_i, 1. - x_i, val_i[rdim]);
    detail_BernsteinBasis::calc_dshape_1d<Float> (p, x_i, 1. - x_i, deriv_i[rdim]);
  }

  // Compute shape values and derivatives for first dimension.
  const Float x_0 = xyz[0];
  detail_BernsteinBasis::calc_shape_1d<Float> (p, x_0, 1. - x_0, val_0);
  detail_BernsteinBasis::calc_dshape_1d<Float> (p, x_0, 1. - x_0, deriv_0);

  //
  // Accumulate the tensor product components.
  // Read each control point once.
  //

  // Set up index formulas.
  // Last coordinate is outermost, encompasses all. First is innermost, encompases (p+1).
  int32 stride[RefDim];
  stride[0] = 1;
  for (int32 rdim = 1; rdim < RefDim; rdim++)
    stride[rdim] = stride[rdim - 1] * (pp1);
  int32 el_dofs = stride[RefDim - 1] * (pp1);

  // Iterate over degrees of freedom, i.e., iterate over control point values.
  for (int32 dof_idx = 0; dof_idx < el_dofs; dof_idx++)
  {
    int32 ii[RefDim];

    Float t_shape_val = 1.;
    Float shape_val_1d[RefDim]; // Cache the values, we'll reuse multiple times in the derivative computation.
    for (int32 rdim_in = 0; rdim_in < RefDim; rdim_in++)
    {
      ii[rdim_in] = (dof_idx / stride[rdim_in]) % (pp1);
      shape_val_1d[rdim_in] = val_i[rdim_in][ii[rdim_in]];
      t_shape_val *= shape_val_1d[rdim_in];
    }

    // Multiply control point value, accumulate value.
    const Vec<Float, PhysDim> ctrl_val = coeff_iter[dof_idx];
    result_val += ctrl_val * t_shape_val;

    for (int32 rdim_out = 0; rdim_out < RefDim; rdim_out++) // Over the derivatives.
    {
      Float t_shape_deriv = 1.;
      int32 rdim_in;
      for (rdim_in = 0; rdim_in < rdim_out; rdim_in++) // Over the tensor dimensions.
        t_shape_deriv *= val_i[rdim_in][ii[rdim_in]];
      t_shape_deriv *= deriv_i[rdim_out][ii[rdim_out]];
      for (++rdim_in; rdim_in < RefDim; rdim_in++) // Over the tensor dimensions.
        t_shape_deriv *= val_i[rdim_in][ii[rdim_in]];

      // Multiply control point value, accumulate value.
      result_deriv[rdim_out] += ctrl_val * t_shape_deriv;
    }
  }
}


template <int32 RefDim>
template <typename CoeffIterType, int32 PhysDim>
DRAY_EXEC Vec<Float, PhysDim>
BernsteinBasis<RefDim>::get_sub_coefficient (const Range *ref_box,
                                             const CoeffIterType &coeff_iter,
                                             int32 p,
                                             int32 i0,
                                             int32 i1,
                                             int32 i2)
{
  // i0...x  i1...y  i2...z
  // assuming coeff iter goes with x on inside.

  // Coordinates of the sub-element box in reference space.
  const Float u0 = (RefDim > 0 ? ref_box[0].min () : 0.0);
  const Float u1 = (RefDim > 0 ? ref_box[0].max () : 0.0);
  const Float v0 = (RefDim > 1 ? ref_box[1].min () : 0.0);
  const Float v1 = (RefDim > 1 ? ref_box[1].max () : 0.0);
  const Float w0 = (RefDim > 2 ? ref_box[2].min () : 0.0);
  const Float w1 = (RefDim > 2 ? ref_box[2].max () : 0.0);

  /// fprintf(stderr, "ref_box==%f %f %f %f %f %f\n",
  ///     u0, u1, v0, v1, w0, w1);

  // Iteration ranges (skip implicit 0s).
  const int32 j0_min = (u1 >= 1.0 ? i0 : 0);
  const int32 j1_min = (v1 >= 1.0 ? i1 : 0);
  const int32 j2_min = (w1 >= 1.0 ? i2 : 0);
  const int32 j0_max = (u0 <= 0.0 ? i0 : p);
  const int32 j1_max = (v0 <= 0.0 ? i1 : p);
  const int32 j2_max = (w0 <= 0.0 ? i2 : p);

  /// #ifdef DEBUG_CPU_ONLY
  ///   fprintf(stderr, "j0:[%d,%d] j1:[%d,%d] j2:[%d,%d]",
  ///       j0_min, j0_max,
  ///       j1_min, j1_max,
  ///       j2_min, j2_max);
  /// #endif

  // Set up matrix columns (if left-multiplying the coefficient list).
  Float W0[MaxPolyOrder + 1];
  if (u0 <= 0.0)
    splitting_matrix_1d_left_seq (p, i0, u1, W0);
  else if (u1 >= 1.0)
    splitting_matrix_1d_right_seq (p, i0, u0, W0 + i0);
  else
    splitting_matrix_1d_seq (p, i0, u0, u1, W0);

  Float W1[MaxPolyOrder + 1];
  if (v0 <= 0.0)
    splitting_matrix_1d_left_seq (p, i1, v1, W1);
  else if (v1 >= 1.0)
    splitting_matrix_1d_right_seq (p, i1, v0, W1 + i1);
  else
    splitting_matrix_1d_seq (p, i1, v0, v1, W1);

  Float W2[MaxPolyOrder + 1];
  if (w0 <= 0.0)
    splitting_matrix_1d_left_seq (p, i2, w1, W2);
  else if (w1 >= 1.0)
    splitting_matrix_1d_right_seq (p, i2, w0, W2 + i2);
  else
    splitting_matrix_1d_seq (p, i2, w0, w1, W2);

  /// #ifdef DEBUG_CPU_ONLY
  ///   fprintf(stderr, "ii:(%d,%d,%d)\tW0:[%f,%f,%f]\tW1:[%f,%f,%f]\tW2:[%f,%f,%f] ",
  ///       i0, i1, i2,
  ///       W0[0], W0[1], W0[2],
  ///       W1[0], W1[1], W1[2],
  ///       W2[0], W2[1], W2[2]);
  /// #endif

  // Product of subdivision weights with original coefficients.
  Vec<Float, PhysDim> new_node;
  new_node = 0.0;
  const int32 s0 = 1;
  const int32 s1 = (p + 1);
  const int32 s2 = (p + 1) * (p + 1);
  for (int32 j2 = j2_min; j2 <= j2_max; j2++)
    for (int32 j1 = j1_min; j1 <= j1_max; j1++)
      for (int32 j0 = j0_min; j0 <= j0_max; j0++)
        new_node = new_node + coeff_iter[j0 * s0 + j1 * s1 + j2 * s2] *
                              (W0[j0] * W1[j1] * W2[j2]);

  return new_node;
}


template <int32 RefDim>
DRAY_EXEC void
BernsteinBasis<RefDim>::splitting_matrix_1d_seq (int32 p, int32 ii, Float t0, Float t1, Float *W)
{
  /// fprintf(stderr, "middle ");
  for (int32 jj = 0; jj <= p; jj++)
    W[jj] = splitting_matrix_1d_comp (p, ii, jj, t0, t1);
}

//
// splitting_matrix_1d_comp()
//
template <int32 RefDim>
DRAY_EXEC Float BernsteinBasis<RefDim>::splitting_matrix_1d_comp (int32 p,
                                                                  int32 ii,
                                                                  int32 jj,
                                                                  Float t0,
                                                                  Float t1)
{
  // Masado Ishii, 2018-06-12, LLNL
  //
  // Derived by applying DeCasteljau twice and collecting weights of each polynomial coefficient.
  //
  //   W = sum(kk=0,min(p-i,p-j)){ beta_{k}^{p-i}(1 - t0/t1) * beta_{j}^{p-k}(t1)
  //
  //   where beta_{j}^{n} is the jth Bernstein basis polynomial of order n.
  //
  // The summation is a convolution, computed by a hybrid of Horner's rule and accumulated powers.

  const int32 ll = p - max (ii, jj);

  BinomRowIterator b1, b2;
  b1.construct (p, jj);
  b2.construct (p - ii, 0);

  // Arguments to the Bernstein basis functions.
  const Float x1 = t1;
  const Float x1bar = 1.0 - x1;
  const Float x2bar = t0 / t1;
  const Float x2 = 1.0 - x2bar;

  // common_factor = pow(x1,j) * pow(1-x1, p-j-l) * pow(1-x2, p-i-l);
  Float common_factor = 1.0;
  {
    int32 power = jj;
    while (power-- > 0)
      common_factor *= x1;
    power = p - jj - ll;
    while (power-- > 0)
      common_factor *= x1bar;
    power = p - ii - ll;
    while (power-- > 0)
      common_factor *= x2bar;
  }

  // Factors for computing the convolution - replaces role of x1, x1bar, x2, x2bar.
  const Float arg_inc = x2; // Accumulate powers of this.
  const Float arg_dec = x1bar * x2bar; // Roll this into each prefix sum (Horner's rule).

  Float arg_inc_pow = 1.0;
  Float W = ((*b1) * (*b2)) * arg_inc_pow;
  for (int32 kk = 1; kk <= ll; kk++)
  {
    arg_inc_pow *= arg_inc;
    b1.lower_n ();
    b2.next ();
    W = W * arg_dec + ((*b1) * (*b2)) * arg_inc_pow;
  }

  W *= common_factor;

  return W;
}

template <int32 RefDim>
DRAY_EXEC void
BernsteinBasis<RefDim>::splitting_matrix_1d_left_seq (int32 p, int32 ii, Float t1, Float *W)
{
  // Masado Ishii, 2018-06-13, LLNL
  //
  // Derived by applying DeCasteljau and collecting weights of each polynomial coefficient.
  //
  //   W = beta_{j}^{i}(t1)
  //
  //   where beta_{j}^{n} is the jth Bernstein basis polynomial of order n.
  //
  // This method is similar to the mfem approach of calc_shape.

  /// fprintf(stderr, "left(%f)  ",t1);
  BinomRowIterator b;
  b.construct (ii, 0);
  Float wpow = 1.0;
  for (int32 jj = 0; jj <= ii; jj++)
  {
    W[jj] = *b * wpow;
    wpow *= t1;
    b.next ();
  }
  wpow = 1 - t1;
  for (int32 jj = 1; jj <= ii; jj++)
  {
    W[ii - jj] *= wpow;
    wpow *= (1 - t1);
  }
}

template <int32 RefDim>
DRAY_EXEC void
BernsteinBasis<RefDim>::splitting_matrix_1d_right_seq (int32 p, int32 ii, Float t0, Float *W)
{
  // splitting_matrix_1d_left_seq(p, p-ii, 1-t0, W[reverse]);

  /// fprintf(stderr, "right  ");
  const Float t1 = 1.0 - t0;
  ii = p - ii;

  BinomRowIterator b;
  b.construct (ii, 0);
  Float wpow = 1.0;
  for (int32 jj = 0; jj <= ii; jj++)
  {
    W[ii - jj] = *b * wpow;
    wpow *= t1;
    b.next ();
  }
  wpow = 1 - t1;
  for (int32 jj = 1; jj <= ii; jj++)
  {
    W[jj] *= wpow;
    wpow *= (1 - t1);
  }
}


} // namespace dray

#endif
