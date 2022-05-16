// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BEZIER_SIMPLEX_HPP
#define DRAY_BEZIER_SIMPLEX_HPP

#include <dray/exports.hpp>
#include <dray/integer_utils.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <typename T, int32 dim> struct BezierSimplex
{
  // BezierSimplex is a template class.
  // Only some instantiations are defined.

  // Generally, the interface is:
  //   DofT value = eval(ref_coords, dof_ptr, poly_order);
  //   DofT value = eval_d(ref_coords, dof_ptr, poly_order, out_derivs);
  //   DofT value = eval_pd(ref_coords, dof_ptr, poly_order, partial_axis, out_deriv);
};


/*
 * Example evaluation using Horner's rule.
 *
 * Dofs for a quadratic triangular element form a triangle.
 *
 * Barycentric coordinates u + v + t = 1.
 *
 * To evaluate an element using Horner's rule, while accessing dofs
 * in the lexicographic order of (u fastest, v next fastest),
 * express the summation as follows:
 *
 *           7  []   \`           v^2 * {  Mck(2;0,2,0)*u^0 }
 *          /         \`
 *    'v'  /           \` (t=0)
 *        /  []    []   \`        v^1 * { [Mck(2;0,1,1)*u^0]*t +
 * Mck(2;1,1,0)*u^1 } /               \` /                 \`
 *        []    []    []          v^0 * { [[Mck(2;0,0,2)*u^0]*t +
 * Mck(2;1,0,1)*u^1]*t + Mck(2;2,0,0)*u^2 } (t=1)
 *        ------------->
 *             'u'
 *
 *  where 'Mck' stands for M choose K, or multinomial coefficient.
 *
 *  Note that multinomial coefficients are symmetric in the indices.
 *  In particular,
 *      Mck(2; 0,0,2) == Mck(2; 2,0,0)
 *      Mck(2; 0,1,1) == Mck(2; 1,1,0)
 *  This property allows us to traverse Pascal's simplex using only
 *  transitions between 'adjacent' multinomial coefficients.
 */


//
// BezierSimplex<T,2> (Bezier triangle).
//
template <typename T> struct BezierSimplex<T, 2>
{
  // eval(): Evaluation without computing derivatives.
  template <typename DofT, typename PtrT = const DofT *>
  DRAY_EXEC static DofT eval (const Vec<T, 2> &ref_coords, PtrT dof_ptr, const int32 p)
  {
    // Barycentric coordinates.
    const T &u = ref_coords[0];
    const T &v = ref_coords[1];
    const T t = T (1.0) - (u + v);

    // Multinomial coefficient. Will traverse Pascal's simplex using
    // transitions between adjacent multinomial coefficients (slide_over()),
    // and transpositions back to the start of each row (swap_places()).
    MultinomialCoeff<2> mck;
    mck.construct (p);

    DofT j_sum;
    j_sum = 0.0;
    T vpow = 1.0;
    for (int32 jj = 0; jj <= p; jj++)
    {

      DofT i_sum;
      i_sum = 0.0;
      T upow = 1.0;
      for (int32 ii = 0; ii <= (p - jj); ii++)
      {
        // Horner's rule innermost, due to decreasing powers of t (mu = p - jj - ii).
        i_sum *= t;
        i_sum = i_sum + (*(dof_ptr++)) * (mck.get_val () * upow);
        upow *= u;
        if (ii < (p - jj)) mck.slide_over (0);
      }
      mck.swap_places (0);

      j_sum = j_sum + i_sum * vpow;
      vpow *= v;
      if (jj < p) mck.slide_over (1);
    }
    // mck.swap_places(1);

    return j_sum;
  }
};


//
// BezierSimplex<T,3> (Bezier tetrahedron).
//
template <typename T> struct BezierSimplex<T, 3>
{
  template <typename DofT, typename PtrT = const DofT *>
  DRAY_EXEC static DofT eval (const Vec<T, 2> &ref_coords, PtrT dof_ptr, const int32 p)
  {
    // TODO
  }
};


} // namespace dray


#endif // DRAY_BEZIER_SIMPLEX_HPP
