// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POS_TENSOR_ELEMENT_TCC
#define DRAY_POS_TENSOR_ELEMENT_TCC

/**
 * @file pos_tensor_element.hpp
 * @brief Partial template specialization of Element_impl
 *        for tensor (i.e. hex and quad) elements.
 */

/// #include <dray/data_model/element.hpp>
#include <dray/integer_utils.hpp> // MultinomialCoeff
#include <dray/vec.hpp>

#include <dray/data_model/bernstein_basis.hpp> // get_sub_coefficient

namespace dray
{



// TODO add get_sub_bounds to each specialization of SubRef.

// ---------------------------------------------------------------------------


// -----
// Implementations
// -----

template <int32 dim>
DRAY_EXEC bool QuadRefSpace<dim>::is_inside (const Vec<Float, dim> &ref_coords)
{
  Float min_val = 2.f;
  Float max_val = -1.f;
  for (int32 d = 0; d < dim; d++)
  {
    min_val = min (ref_coords[d], min_val);
    max_val = max (ref_coords[d], max_val);
  }
  return (min_val >= 0.f - epsilon<Float> ()) && (max_val <= 1.f + epsilon<Float> ());
}


template <int32 dim>
DRAY_EXEC bool QuadRefSpace<dim>::is_inside (const Vec<Float, dim> &ref_coords,
                                             const Float &eps)
{
  Float min_val = 2.f;
  Float max_val = -1.f;
  for (int32 d = 0; d < dim; d++)
  {
    min_val = min (ref_coords[d], min_val);
    max_val = max (ref_coords[d], max_val);
  }
  return (min_val >= 0.f - eps) && (max_val <= 1.f + eps);
}


// ---------------------------------------------------------------------------

// Template specialization (Tensor type, general order).
//
// Assume dim <= 3.
//
template <int32 dim, int32 ncomp>
class Element_impl<dim, ncomp, ElemType::Tensor, Order::General> : public QuadRefSpace<dim>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;
  uint32 m_order;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 poly_order)
  {
    m_dof_ptr = dof_ptr;
    m_order = poly_order;
  }
  DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>> read_dof_ptr() const
  {
    return m_dof_ptr;
  }
  DRAY_EXEC int32 get_order () const
  {
    return m_order;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, dim> &ref_coords) const
  {
    //TODO make separate eval() and don't call eval_d().
    Vec<Vec<Float, ncomp>, dim> unused_deriv;
    return eval_d(ref_coords, unused_deriv);
  }

  DRAY_EXEC Vec<Float, ncomp> eval_d (const Vec<Float, dim> &ref_coords,
                                      Vec<Vec<Float, ncomp>, dim> &out_derivs) const
  {
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

    using DofT = Vec<Float, ncomp>;

    DofT zero;
    zero = 0;

    const Float u = (dim > 0 ? ref_coords[0] : 0.0);
    const Float v = (dim > 1 ? ref_coords[1] : 0.0);
    const Float w = (dim > 2 ? ref_coords[2] : 0.0);
    const Float ubar = 1.0 - u;
    const Float vbar = 1.0 - v;
    const Float wbar = 1.0 - w;

    const int32 p1 = (dim >= 1 ? m_order : 0);
    const int32 p2 = (dim >= 2 ? m_order : 0);
    const int32 p3 = (dim >= 3 ? m_order : 0);

    combo_int B[MaxPolyOrder];
    if (m_order >= 1)
    {
      BinomialCoeff binomial_coeff;
      binomial_coeff.construct (m_order - 1);
      for (int32 ii = 0; ii <= m_order - 1; ii++)
      {
        B[ii] = binomial_coeff.get_val ();
        binomial_coeff.slide_over (0);
      }
    }

    int32 cidx = 0; // Index into element dof indexing space.

    // Compute and combine order (p-1) values to get order (p) values/derivatives.
    // https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties

    DofT val_u, val_v, val_w;
    DofT deriv_u;
    Vec<DofT, 2> deriv_uv;
    Vec<DofT, 3> deriv_uvw;

    // Level3 set up.
    Float wpow = 1.0;
    Vec<DofT, 3> val_w_L, val_w_R; // Second/third columns are derivatives in lower level.
    val_w_L = zero;
    val_w_R = zero;
    for (int32 ii = 0; ii <= p3; ii++)
    {
      // Level2 set up.
      Float vpow = 1.0;
      Vec<DofT, 2> val_v_L, val_v_R; // Second column is derivative in lower level.
      val_v_L = zero;
      val_v_R = zero;

      for (int32 jj = 0; jj <= p2; jj++)
      {
        // Level1 set up.
        Float upow = 1.0;
        DofT val_u_L = zero, val_u_R = zero; // L and R can be combined --> val, deriv.
        DofT C = m_dof_ptr[cidx++];
        for (int32 kk = 1; kk <= p1; kk++)
        {
          // Level1 accumulation.
          val_u_L = val_u_L * ubar + C * (B[kk - 1] * upow);
          C = m_dof_ptr[cidx++];
          val_u_R = val_u_R * ubar + C * (B[kk - 1] * upow);
          upow *= u;
        } // kk

        // Level1 result.
        val_u = (p1 > 0 ? val_u_L * ubar + val_u_R * u : C);
        deriv_u = (val_u_R - val_u_L) * p1;

        // Level2 accumulation.
        if (jj > 0)
        {
          val_v_R[0] = val_v_R[0] * vbar + val_u * (B[jj - 1] * vpow);
          val_v_R[1] = val_v_R[1] * vbar + deriv_u * (B[jj - 1] * vpow);
          vpow *= v;
        }
        if (jj < p2)
        {
          val_v_L[0] = val_v_L[0] * vbar + val_u * (B[jj] * vpow);
          val_v_L[1] = val_v_L[1] * vbar + deriv_u * (B[jj] * vpow);
        }
      } // jj

      // Level2 result.
      val_v = (p2 > 0 ? val_v_L[0] * vbar + val_v_R[0] * v : val_u);
      deriv_uv[0] = (p2 > 0 ? val_v_L[1] * vbar + val_v_R[1] * v : deriv_u);
      deriv_uv[1] = (val_v_R[0] - val_v_L[0]) * p2;

      // Level3 accumulation.
      if (ii > 0)
      {
        val_w_R[0] = val_w_R[0] * wbar + val_v * (B[ii - 1] * wpow);
        val_w_R[1] = val_w_R[1] * wbar + deriv_uv[0] * (B[ii - 1] * wpow);
        val_w_R[2] = val_w_R[2] * wbar + deriv_uv[1] * (B[ii - 1] * wpow);
        wpow *= w;
      }
      if (ii < p3)
      {
        val_w_L[0] = val_w_L[0] * wbar + val_v * (B[ii] * wpow);
        val_w_L[1] = val_w_L[1] * wbar + deriv_uv[0] * (B[ii] * wpow);
        val_w_L[2] = val_w_L[2] * wbar + deriv_uv[1] * (B[ii] * wpow);
      }
    } // ii

    // Level3 result.
    val_w = (p3 > 0 ? val_w_L[0] * wbar + val_w_R[0] * w : val_v);
    deriv_uvw[0] = (p3 > 0 ? val_w_L[1] * wbar + val_w_R[1] * w : deriv_uv[0]);
    deriv_uvw[1] = (p3 > 0 ? val_w_L[2] * wbar + val_w_R[2] * w : deriv_uv[1]);
    deriv_uvw[2] = (val_w_R[0] - val_w_L[0]) * p3;

    if (dim > 0) out_derivs[0] = deriv_uvw[0];
    if (dim > 1) out_derivs[1] = deriv_uvw[1];
    if (dim > 2) out_derivs[2] = deriv_uvw[2];

    return val_w;
  }

  DRAY_EXEC void get_sub_bounds (const SubRef<dim, ElemType::Tensor> &sub_ref, AABB<ncomp> &aabb) const;
};


//
// get_sub_bounds()
template <int32 dim, int32 ncomp>
DRAY_EXEC void
Element_impl<dim, ncomp, ElemType::Tensor, Order::General>::get_sub_bounds (const SubRef<dim, ElemType::Tensor> &sub_ref,
                                                                          AABB<ncomp> &aabb) const
{
  // Initialize.
  aabb.reset ();
  OrderPolicy<General> policy;
  policy.value = get_order();

  const int32 num_dofs = eattr::get_num_dofs ( Shape<dim, Tensor>{},
                                               policy);

  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  if (m_order <= 3) // TODO find the optimal threshold, if there is one.
  {
    // Get the sub-coefficients all at once in a block.
    switch (m_order)
    {
    case 1:
    {
      constexpr int32 POrder = 1;
      AABB<dim> subref_aabb;
      subref_aabb.include(sub_ref[0]);
      subref_aabb.include(sub_ref[1]);
      MultiVec<Float, dim, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<dim, ncomp, POrder, PtrT> (subref_aabb.m_ranges, m_dof_ptr);
      for (int32 ii = 0; ii < num_dofs; ii++)
        aabb.include (sub_nodes.linear_idx (ii));
    }
    break;

    case 2:
    {
      constexpr int32 POrder = 2;
      AABB<dim> subref_aabb;
      subref_aabb.include(sub_ref[0]);
      subref_aabb.include(sub_ref[1]);
      MultiVec<Float, dim, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<dim, ncomp, POrder, PtrT> (subref_aabb.m_ranges, m_dof_ptr);
      for (int32 ii = 0; ii < num_dofs; ii++)
        aabb.include (sub_nodes.linear_idx (ii));
    }
    break;

    case 3:
    {
      constexpr int32 POrder = 3;
      AABB<dim> subref_aabb;
      subref_aabb.include(sub_ref[0]);
      subref_aabb.include(sub_ref[1]);
      MultiVec<Float, dim, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<dim, ncomp, POrder, PtrT> (subref_aabb.m_ranges, m_dof_ptr);
      for (int32 ii = 0; ii < num_dofs; ii++)
        aabb.include (sub_nodes.linear_idx (ii));
    }
    break;
    }
  }
  else
  {
    AABB<dim> subref_aabb;
    subref_aabb.include(sub_ref[0]);
    subref_aabb.include(sub_ref[1]);

    // Get each sub-coefficient one at a time.
    for (int32 i0 = 0; i0 <= (dim >= 1 ? m_order : 0); i0++)
      for (int32 i1 = 0; i1 <= (dim >= 2 ? m_order : 0); i1++)
        for (int32 i2 = 0; i2 <= (dim >= 3 ? m_order : 0); i2++)
        {
          Vec<Float, ncomp> sub_node =
          // TODO move out of bernstein_basis.hpp
          BernsteinBasis<dim>::template get_sub_coefficient<PtrT, ncomp> (
          subref_aabb.m_ranges, m_dof_ptr, m_order, i0, i1, i2);
          aabb.include (sub_node);
        }
  }
}


// ---------------------------------------------------------------------------


// Template specialization (Tensor type, 0th order).
//
template <int32 dim, int32 ncomp>
class Element_impl<dim, ncomp, ElemType::Tensor, Order::Constant> : public QuadRefSpace<dim>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>> read_dof_ptr() const
  {
    return m_dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 0;
  }

  // Get value without derivative.
  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, dim> &ref_coords) const
  {
    return *m_dof_ptr;
  }

  // Get value with derivative.
  DRAY_EXEC Vec<Float, ncomp> eval_d (const Vec<Float, dim> &ref_coords,
                                      Vec<Vec<Float, ncomp>, dim> &out_derivs) const
  {
    for (int d = 0; d < dim; d++)
      out_derivs[d] = 0;

    return *m_dof_ptr;
  }

  DRAY_EXEC void get_sub_bounds(const SubRef<dim, ElemType::Tensor> &sub_ref, AABB<ncomp> &aabb) const
  {
    aabb.reset ();
    aabb.include (m_dof_ptr[0]);
  }
};


// Template specialization (Tensor type, 1st order, 2D).
//
template <int32 ncomp>
class Element_impl<2u, ncomp, ElemType::Tensor, Order::Linear> : public QuadRefSpace<2u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>> read_dof_ptr() const
  {
    return m_dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 1;
  }

  // Get value without derivative.
  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 2u> &r) const
  {
    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) + m_dof_ptr[1] * r[0] * (1 - r[1]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] + m_dof_ptr[3] * r[0] * r[1];
  }

  // Get value with derivative.
  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 2u> &r, Vec<Vec<Float, ncomp>, 2u> &out_derivs) const
  {
    out_derivs[0] = (m_dof_ptr[1] - m_dof_ptr[0]) * (1 - r[1]) +
                    (m_dof_ptr[3] - m_dof_ptr[2]) * r[1];

    out_derivs[1] = (m_dof_ptr[2] - m_dof_ptr[0]) * (1 - r[0]) +
                    (m_dof_ptr[3] - m_dof_ptr[1]) * r[0];

    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) + m_dof_ptr[1] * r[0] * (1 - r[1]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] + m_dof_ptr[3] * r[0] * r[1];
  }

  DRAY_EXEC void get_sub_bounds(const SubRef<2, ElemType::Tensor> &sub_ref, AABB<ncomp> &aabb) const
  {
//#ifndef NDEBUG
//#warning "Tensor element linear 2D get_sub_bounds() returns full bounds, don't use. (but could implement)"
//#endif
    aabb.reset ();
    const int num_dofs = eattr::get_num_dofs (ShapeQuad{}, OrderPolicy<Linear>{});
    for (int ii = 0; ii < num_dofs; ii++)
      aabb.include (m_dof_ptr[ii]);
  }
};


// Template specialization (Tensor type, 1st order, 3D).
//
template <int32 ncomp>
class Element_impl<3u, ncomp, ElemType::Tensor, Order::Linear> : public QuadRefSpace<3u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>> read_dof_ptr() const
  {
    return m_dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 1;
  }

  DRAY_EXEC void get_sub_bounds (const SubRef<3, ElemType::Tensor> &sub_ref, AABB<ncomp> &aabb) const
  {
    using PtrT = SharedDofPtr<Vec<Float, ncomp>>;
    constexpr int32 POrder = 1;
    AABB<3> subref_aabb;
    subref_aabb.include(sub_ref[0]);
    subref_aabb.include(sub_ref[1]);
    MultiVec<Float, 3u, ncomp, POrder> sub_nodes =
      sub_element_fixed_order<3, ncomp, POrder, PtrT> (subref_aabb.m_ranges, m_dof_ptr);
    for (int32 ii = 0; ii < 8; ii++)
       aabb.include (sub_nodes.linear_idx (ii));
  }

  // Get value without derivative.
  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 3u> &r) const
  {
    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[1] * r[0] * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] * (1 - r[2]) +
           m_dof_ptr[3] * r[0] * r[1] * (1 - r[2]) +
           m_dof_ptr[4] * (1 - r[0]) * (1 - r[1]) * r[2] +
           m_dof_ptr[5] * r[0] * (1 - r[1]) * r[2] +
           m_dof_ptr[6] * (1 - r[0]) * r[1] * r[2] + m_dof_ptr[7] * r[0] * r[1] * r[2];
  }

  // Get value with derivative.
  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 3u> &r, Vec<Vec<Float, ncomp>, 3u> &out_derivs) const
  {
    out_derivs[0] = (m_dof_ptr[1] - m_dof_ptr[0]) * (1 - r[1]) * (1 - r[2]) +
                    (m_dof_ptr[3] - m_dof_ptr[2]) * r[1] * (1 - r[2]) +
                    (m_dof_ptr[5] - m_dof_ptr[4]) * (1 - r[1]) * r[2] +
                    (m_dof_ptr[7] - m_dof_ptr[6]) * r[1] * r[2];

    out_derivs[1] = (m_dof_ptr[2] - m_dof_ptr[0]) * (1 - r[0]) * (1 - r[2]) +
                    (m_dof_ptr[3] - m_dof_ptr[1]) * r[0] * (1 - r[2]) +
                    (m_dof_ptr[6] - m_dof_ptr[4]) * (1 - r[0]) * r[2] +
                    (m_dof_ptr[7] - m_dof_ptr[5]) * r[0] * r[2];

    out_derivs[2] = (m_dof_ptr[4] - m_dof_ptr[0]) * (1 - r[0]) * (1 - r[1]) +
                    (m_dof_ptr[5] - m_dof_ptr[1]) * r[0] * (1 - r[1]) +
                    (m_dof_ptr[6] - m_dof_ptr[2]) * (1 - r[0]) * r[1] +
                    (m_dof_ptr[7] - m_dof_ptr[3]) * r[0] * r[1];

    return m_dof_ptr[0] * (1 - r[0]) * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[1] * r[0] * (1 - r[1]) * (1 - r[2]) +
           m_dof_ptr[2] * (1 - r[0]) * r[1] * (1 - r[2]) +
           m_dof_ptr[3] * r[0] * r[1] * (1 - r[2]) +
           m_dof_ptr[4] * (1 - r[0]) * (1 - r[1]) * r[2] +
           m_dof_ptr[5] * r[0] * (1 - r[1]) * r[2] +
           m_dof_ptr[6] * (1 - r[0]) * r[1] * r[2] + m_dof_ptr[7] * r[0] * r[1] * r[2];
  }
};


// Template specialization (Tensor type, 2nd order, 2D).
//
template <int32 ncomp>
class Element_impl<2u, ncomp, ElemType::Tensor, Order::Quadratic> : public QuadRefSpace<2u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>> read_dof_ptr() const
  {
    return m_dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 2;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 2u> &r) const
  {
    // Shape functions. Quadratic has 3 1D shape functions on each axis.
    Float su[3] = { (1 - r[0]) * (1 - r[0]), 2 * r[0] * (1 - r[0]), r[0] * r[0] };
    Float sv[3] = { (1 - r[1]) * (1 - r[1]), 2 * r[1] * (1 - r[1]), r[1] * r[1] };

    return m_dof_ptr[0] * su[0] * sv[0] + m_dof_ptr[1] * su[1] * sv[0] +
           m_dof_ptr[2] * su[2] * sv[0] + m_dof_ptr[3] * su[0] * sv[1] +
           m_dof_ptr[4] * su[1] * sv[1] + m_dof_ptr[5] * su[2] * sv[1] +
           m_dof_ptr[6] * su[0] * sv[2] + m_dof_ptr[7] * su[1] * sv[2] +
           m_dof_ptr[8] * su[2] * sv[2];
  }

  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 2u> &r, Vec<Vec<Float, ncomp>, 2u> &out_derivs) const
  {
    // Shape functions. Quadratic has 3 1D shape functions on each axis.
    Float su[3] = { (1 - r[0]) * (1 - r[0]), 2 * r[0] * (1 - r[0]), r[0] * r[0] };
    Float sv[3] = { (1 - r[1]) * (1 - r[1]), 2 * r[1] * (1 - r[1]), r[1] * r[1] };

    // Shape derivatives.
    Float dsu[3] = { -2*(1-r[0]), 2 - 4*r[0], 2*r[0] };
    Float dsv[3] = { -2*(1-r[1]), 2 - 4*r[1], 2*r[1] };

    out_derivs[0] = m_dof_ptr[0] * dsu[0] * sv[0] +
                    m_dof_ptr[1] * dsu[1] * sv[0] + m_dof_ptr[2] * dsu[2] * sv[0] +
                    m_dof_ptr[3] * dsu[0] * sv[1] + m_dof_ptr[4] * dsu[1] * sv[1] +
                    m_dof_ptr[5] * dsu[2] * sv[1] + m_dof_ptr[6] * dsu[0] * sv[2] +
                    m_dof_ptr[7] * dsu[1] * sv[2] + m_dof_ptr[8] * dsu[2] * sv[2];

    out_derivs[1] = m_dof_ptr[0] * su[0] * dsv[0] +
                    m_dof_ptr[1] * su[1] * dsv[0] + m_dof_ptr[2] * su[2] * dsv[0] +
                    m_dof_ptr[3] * su[0] * dsv[1] + m_dof_ptr[4] * su[1] * dsv[1] +
                    m_dof_ptr[5] * su[2] * dsv[1] + m_dof_ptr[6] * su[0] * dsv[2] +
                    m_dof_ptr[7] * su[1] * dsv[2] + m_dof_ptr[8] * su[2] * dsv[2];

    return m_dof_ptr[0] * su[0] * sv[0] + m_dof_ptr[1] * su[1] * sv[0] +
           m_dof_ptr[2] * su[2] * sv[0] + m_dof_ptr[3] * su[0] * sv[1] +
           m_dof_ptr[4] * su[1] * sv[1] + m_dof_ptr[5] * su[2] * sv[1] +
           m_dof_ptr[6] * su[0] * sv[2] + m_dof_ptr[7] * su[1] * sv[2] +
           m_dof_ptr[8] * su[2] * sv[2];
  }

  DRAY_EXEC void get_sub_bounds(const SubRef<2, ElemType::Tensor> &sub_ref, AABB<ncomp> &aabb) const
  {
//#ifndef NDEBUG
//#warning "Tensor element quadratic 2D get_sub_bounds() returns full bounds, don't use."
//#endif
    aabb.reset ();
    const int num_dofs = eattr::get_num_dofs (ShapeQuad{}, OrderPolicy<Quadratic>{});
    for (int ii = 0; ii < num_dofs; ii++)
      aabb.include (m_dof_ptr[ii]);
  }
};


// Template specialization (Tensor type, 2nd order, 3D).
//
template <int32 ncomp>
class Element_impl<3u, ncomp, ElemType::Tensor, Order::Quadratic> : public QuadRefSpace<3u>
{
  protected:
  SharedDofPtr<Vec<Float, ncomp>> m_dof_ptr;

  public:
  DRAY_EXEC void construct (SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
  {
    m_dof_ptr = dof_ptr;
  }
  DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>> read_dof_ptr() const
  {
    return m_dof_ptr;
  }
  DRAY_EXEC static constexpr int32 get_order ()
  {
    return 2;
  }

  DRAY_EXEC Vec<Float, ncomp> eval (const Vec<Float, 3> &ref_coords) const
  {
    //TODO make separate eval() and don't call eval_d().
    Vec<Vec<Float, ncomp>, 3> unused_deriv;
    return eval_d(ref_coords, unused_deriv);
  }

  DRAY_EXEC Vec<Float, ncomp>
  eval_d (const Vec<Float, 3u> &r, Vec<Vec<Float, ncomp>, 3u> &out_derivs) const
  {
    // Shape functions. Quadratic has 3 1D shape functions on each axis.
    Float su[3] = { (1 - r[0]) * (1 - r[0]), 2 * r[0] * (1 - r[0]), r[0] * r[0] };
    Float sv[3] = { (1 - r[1]) * (1 - r[1]), 2 * r[1] * (1 - r[1]), r[1] * r[1] };
    Float sw[3] = { (1 - r[2]) * (1 - r[2]), 2 * r[2] * (1 - r[2]), r[2] * r[2] };

    // Shape derivatives.
    Float dsu[3] = { -2*(1-r[0]), 2 - 4*r[0], 2*r[0] };
    Float dsv[3] = { -2*(1-r[1]), 2 - 4*r[1], 2*r[1] };
    Float dsw[3] = { -2*(1-r[2]), 2 - 4*r[2], 2*r[2] };

    out_derivs[0] =
    m_dof_ptr[0] * dsu[0] * sv[0] * sw[0] +
    m_dof_ptr[1] * dsu[1] * sv[0] * sw[0] + m_dof_ptr[2] * dsu[2] * sv[0] * sw[0] +
    m_dof_ptr[3] * dsu[0] * sv[1] * sw[0] + m_dof_ptr[4] * dsu[1] * sv[1] * sw[0] +
    m_dof_ptr[5] * dsu[2] * sv[1] * sw[0] + m_dof_ptr[6] * dsu[0] * sv[2] * sw[0] +
    m_dof_ptr[7] * dsu[1] * sv[2] * sw[0] + m_dof_ptr[8] * dsu[2] * sv[2] * sw[0] +

    m_dof_ptr[9] * dsu[0] * sv[0] * sw[1] + m_dof_ptr[10] * dsu[1] * sv[0] * sw[1] +
    m_dof_ptr[11] * dsu[2] * sv[0] * sw[1] +
    m_dof_ptr[12] * dsu[0] * sv[1] * sw[1] + m_dof_ptr[13] * dsu[1] * sv[1] * sw[1] +
    m_dof_ptr[14] * dsu[2] * sv[1] * sw[1] + m_dof_ptr[15] * dsu[0] * sv[2] * sw[1] +
    m_dof_ptr[16] * dsu[1] * sv[2] * sw[1] + m_dof_ptr[17] * dsu[2] * sv[2] * sw[1] +

    m_dof_ptr[18] * dsu[0] * sv[0] * sw[2] +
    m_dof_ptr[19] * dsu[1] * sv[0] * sw[2] + m_dof_ptr[20] * dsu[2] * sv[0] * sw[2] +
    m_dof_ptr[21] * dsu[0] * sv[1] * sw[2] + m_dof_ptr[22] * dsu[1] * sv[1] * sw[2] +
    m_dof_ptr[23] * dsu[2] * sv[1] * sw[2] + m_dof_ptr[24] * dsu[0] * sv[2] * sw[2] +
    m_dof_ptr[25] * dsu[1] * sv[2] * sw[2] + m_dof_ptr[26] * dsu[2] * sv[2] * sw[2];

    out_derivs[1] =
    m_dof_ptr[0] * su[0] * dsv[0] * sw[0] +
    m_dof_ptr[1] * su[1] * dsv[0] * sw[0] + m_dof_ptr[2] * su[2] * dsv[0] * sw[0] +
    m_dof_ptr[3] * su[0] * dsv[1] * sw[0] + m_dof_ptr[4] * su[1] * dsv[1] * sw[0] +
    m_dof_ptr[5] * su[2] * dsv[1] * sw[0] + m_dof_ptr[6] * su[0] * dsv[2] * sw[0] +
    m_dof_ptr[7] * su[1] * dsv[2] * sw[0] + m_dof_ptr[8] * su[2] * dsv[2] * sw[0] +

    m_dof_ptr[9] * su[0] * dsv[0] * sw[1] + m_dof_ptr[10] * su[1] * dsv[0] * sw[1] +
    m_dof_ptr[11] * su[2] * dsv[0] * sw[1] +
    m_dof_ptr[12] * su[0] * dsv[1] * sw[1] + m_dof_ptr[13] * su[1] * dsv[1] * sw[1] +
    m_dof_ptr[14] * su[2] * dsv[1] * sw[1] + m_dof_ptr[15] * su[0] * dsv[2] * sw[1] +
    m_dof_ptr[16] * su[1] * dsv[2] * sw[1] + m_dof_ptr[17] * su[2] * dsv[2] * sw[1] +

    m_dof_ptr[18] * su[0] * dsv[0] * sw[2] +
    m_dof_ptr[19] * su[1] * dsv[0] * sw[2] + m_dof_ptr[20] * su[2] * dsv[0] * sw[2] +
    m_dof_ptr[21] * su[0] * dsv[1] * sw[2] + m_dof_ptr[22] * su[1] * dsv[1] * sw[2] +
    m_dof_ptr[23] * su[2] * dsv[1] * sw[2] + m_dof_ptr[24] * su[0] * dsv[2] * sw[2] +
    m_dof_ptr[25] * su[1] * dsv[2] * sw[2] + m_dof_ptr[26] * su[2] * dsv[2] * sw[2];

    out_derivs[2] =
    m_dof_ptr[0] * su[0] * sv[0] * dsw[0] +
    m_dof_ptr[1] * su[1] * sv[0] * dsw[0] + m_dof_ptr[2] * su[2] * sv[0] * dsw[0] +
    m_dof_ptr[3] * su[0] * sv[1] * dsw[0] + m_dof_ptr[4] * su[1] * sv[1] * dsw[0] +
    m_dof_ptr[5] * su[2] * sv[1] * dsw[0] + m_dof_ptr[6] * su[0] * sv[2] * dsw[0] +
    m_dof_ptr[7] * su[1] * sv[2] * dsw[0] + m_dof_ptr[8] * su[2] * sv[2] * dsw[0] +

    m_dof_ptr[9] * su[0] * sv[0] * dsw[1] + m_dof_ptr[10] * su[1] * sv[0] * dsw[1] +
    m_dof_ptr[11] * su[2] * sv[0] * dsw[1] +
    m_dof_ptr[12] * su[0] * sv[1] * dsw[1] + m_dof_ptr[13] * su[1] * sv[1] * dsw[1] +
    m_dof_ptr[14] * su[2] * sv[1] * dsw[1] + m_dof_ptr[15] * su[0] * sv[2] * dsw[1] +
    m_dof_ptr[16] * su[1] * sv[2] * dsw[1] + m_dof_ptr[17] * su[2] * sv[2] * dsw[1] +

    m_dof_ptr[18] * su[0] * sv[0] * dsw[2] +
    m_dof_ptr[19] * su[1] * sv[0] * dsw[2] + m_dof_ptr[20] * su[2] * sv[0] * dsw[2] +
    m_dof_ptr[21] * su[0] * sv[1] * dsw[2] + m_dof_ptr[22] * su[1] * sv[1] * dsw[2] +
    m_dof_ptr[23] * su[2] * sv[1] * dsw[2] + m_dof_ptr[24] * su[0] * sv[2] * dsw[2] +
    m_dof_ptr[25] * su[1] * sv[2] * dsw[2] + m_dof_ptr[26] * su[2] * sv[2] * dsw[2];

    return m_dof_ptr[0] * su[0] * sv[0] * sw[0] +
           m_dof_ptr[1] * su[1] * sv[0] * sw[0] + m_dof_ptr[2] * su[2] * sv[0] * sw[0] +
           m_dof_ptr[3] * su[0] * sv[1] * sw[0] + m_dof_ptr[4] * su[1] * sv[1] * sw[0] +
           m_dof_ptr[5] * su[2] * sv[1] * sw[0] + m_dof_ptr[6] * su[0] * sv[2] * sw[0] +
           m_dof_ptr[7] * su[1] * sv[2] * sw[0] + m_dof_ptr[8] * su[2] * sv[2] * sw[0] +

           m_dof_ptr[9] * su[0] * sv[0] * sw[1] + m_dof_ptr[10] * su[1] * sv[0] * sw[1] +
           m_dof_ptr[11] * su[2] * sv[0] * sw[1] +
           m_dof_ptr[12] * su[0] * sv[1] * sw[1] +
           m_dof_ptr[13] * su[1] * sv[1] * sw[1] +
           m_dof_ptr[14] * su[2] * sv[1] * sw[1] +
           m_dof_ptr[15] * su[0] * sv[2] * sw[1] +
           m_dof_ptr[16] * su[1] * sv[2] * sw[1] +
           m_dof_ptr[17] * su[2] * sv[2] * sw[1] +

           m_dof_ptr[18] * su[0] * sv[0] * sw[2] +
           m_dof_ptr[19] * su[1] * sv[0] * sw[2] +
           m_dof_ptr[20] * su[2] * sv[0] * sw[2] +
           m_dof_ptr[21] * su[0] * sv[1] * sw[2] +
           m_dof_ptr[22] * su[1] * sv[1] * sw[2] +
           m_dof_ptr[23] * su[2] * sv[1] * sw[2] +
           m_dof_ptr[24] * su[0] * sv[2] * sw[2] +
           m_dof_ptr[25] * su[1] * sv[2] * sw[2] +
           m_dof_ptr[26] * su[2] * sv[2] * sw[2];
  }

  DRAY_EXEC void get_sub_bounds(const SubRef<3, ElemType::Tensor> &sub_ref, AABB<ncomp> &aabb) const
  {
//#ifndef NDEBUG
//#warning "Tensor element quadratic 3D get_sub_bounds() returns full bounds, don't use."
//#endif
    aabb.reset ();
    const int num_dofs = eattr::get_num_dofs (ShapeHex{}, OrderPolicy<Quadratic>{});
    for (int ii = 0; ii < num_dofs; ii++)
      aabb.include (m_dof_ptr[ii]);
  }
};



} // namespace dray

#endif // DRAY_POS_TENSOR_ELEMENT_TCC
