// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ELEMENT_HPP
#define DRAY_ELEMENT_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/data_model/bernstein_basis.hpp>
#include <dray/data_model/subpatch.hpp>
#include <dray/data_model/dof_access.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/subref.hpp>
#include <dray/data_model/ref_space.hpp>
#include <dray/aabb.hpp>
#include <dray/range.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

#include <dray/newton_solver.hpp>
#include <dray/subdivision_search.hpp>

namespace dray
{
template <int32 dim, int32 ncomp, ElemType etype, int32 P>
class Element;

// Utility to write an offsets array for a list of non-shared dofs. TODO move out of element.hpp
DRAY_EXEC void init_counting (int32 *offsets_array, int32 size)
{
  for (int32 ii = 0; ii < size; ii++)
    *(offsets_array++) = ii;
}



// ===========================================
// Adapters Element<> --> Shape, OrderPolicy
// ===========================================
// Before we destroy and rebuild Element
// in favor of Shape, OrderPolicy, etc., use this adapter

/** AdaptGetShape::type, adapt_get_shape() */
template <class ElemT>
struct AdaptGetShape { };

template <int32 ncomp, int32 P>
struct AdaptGetShape<Element<2, ncomp, Simplex, P>> { using type = ShapeTri; };

template <int32 ncomp, int32 P>
struct AdaptGetShape<Element<2, ncomp, Tensor, P>> { using type = ShapeQuad; };

template <int32 ncomp, int32 P>
struct AdaptGetShape<Element<3, ncomp, Simplex, P>> { using type = ShapeTet; };

template <int32 ncomp, int32 P>
struct AdaptGetShape<Element<3, ncomp, Tensor, P>> { using type = ShapeHex; };

template <class ElemT>
DRAY_EXEC constexpr typename AdaptGetShape<ElemT>::type adapt_get_shape(const ElemT &)
{
  return typename AdaptGetShape<ElemT>::type{};
}

template <class ElemT>
DRAY_EXEC constexpr typename AdaptGetShape<ElemT>::type adapt_get_shape()  // adapt_get_shape<ElemT>()
{
  return typename AdaptGetShape<ElemT>::type{};
}


/** AdaptGetOrderPolicy::type, adapt_get_order_policy() */
template <class ElemT>
struct AdaptGetOrderPolicy { };

template <int32 dim, int32 ncomp, ElemType etype, int32 P>
struct AdaptGetOrderPolicy<Element<dim, ncomp, etype, P>> { using type = OrderPolicy<P>; };

template <int32 dim, int32 ncomp, ElemType etype, int32 P>
DRAY_EXEC OrderPolicy<P> adapt_get_order_policy(const Element<dim, ncomp, etype, P> &,
                                      const int32)
{
  return OrderPolicy<P>{};
}

template <int32 dim, int32 ncomp, ElemType etype>
DRAY_EXEC OrderPolicy<General> adapt_get_order_policy(const Element<dim, ncomp, etype, General> &,
                                            const int32 order)
{
  return OrderPolicy<General>{order};
}

// ===========================================


template <int32 dim, int32 ncomp, ElemType etype, int32 P = Order::General>
class Element_impl
{
  public:
  // These member functions should be treated as pure virtual.
  /// DRAY_EXEC void construct(SharedDofPtr<Vec<T, ncomp>> dof_ptr, int32 poly_order); //=0
  /// DRAY_EXEC SharedDofPtr<Vec<T, ncomp>> read_dof_ptr() const;
  /// DRAY_EXEC int32 get_order() const;  //=0
  /// DRAY_EXEC Vec<T, ncomp> eval(const Vec<T,dim> &ref_coords) const;  //=0
  /// DRAY_EXEC Vec<T, ncomp> eval_d( const Vec<T,dim> &ref_coords, Vec<Vec<T,ncomp>,dim> &out_derivs) const;  //=0
  /// DRAY_EXEC void get_sub_bounds(const SubRef<dim,etype> &sub_ref, AABB<ncomp> &aabb) const;  //=0
  /// DRAY_EXEC static bool is_inside(const Vec<T,dim> &ref_coords);  //=0
  /// DRAY_EXEC static void clamp_to_domain(Vec<T,dim> &ref_coords);  //=0
  /// DRAY_EXEC static Vec<T,dim> project_to_domain(const Vec<T,dim> &r1, const Vec<T,dim> &r2);  //=0
};
// Several specialization in other files.
// See pos_tensor_element.tcc and pos_simplex_element.tcc
// which are included at the end of the current header file.

template <int32 dim, ElemType etype, int32 P = Order::General>
class InvertibleElement_impl : public Element_impl<dim, dim, etype, P>
{
  public:
  /// //
  /// // eval_inverse() : Try to locate the point in reference space. Return false if not contained.
  /// //
  /// // use_init_guess determines whether guess_domain is used or replaced by AABB::ref_universe().
  /// DRAY_EXEC bool eval_inverse (const Vec<Float, dim> &world_coords,
  ///                              const SubRef<dim, etype> &guess_domain,
  ///                              Vec<Float, dim> &ref_coords,
  ///                              bool use_init_guess = false) const;

  /// DRAY_EXEC bool eval_inverse (stats::Stats &stats,
  ///                              const Vec<Float, dim> &world_coords,
  ///                              const SubRef<dim, etype> &guess_domain,
  ///                              Vec<Float, dim> &ref_coords,
  ///                              bool use_init_guess = false) const;

  DRAY_EXEC bool eval_inverse_local (const Vec<Float, dim> &world_coords,
                                     Vec<Float, dim> &ref_coords) const;

  DRAY_EXEC bool eval_inverse_local (stats::Stats &stats,
                                     const Vec<Float, dim> &world_coords,
                                     Vec<Float, dim> &ref_coords) const;
};


namespace detail
{
//
// positive_get_bounds
//
// In positive bases, function on reference domain is bounded by convex hull of dofs.
template <int32 ncomp>
DRAY_EXEC void
positive_get_bounds (AABB<ncomp> &aabb, SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 num_dofs)
{
  aabb.reset ();
  while (num_dofs--)
  {
    aabb.include (*dof_ptr);
    ++dof_ptr;
  }
}

} // namespace detail


// =========================================
// Element<> Wrapper Interface
// =========================================


/**
 * @tparam dim Topological dimension, i.e. dimensionality of reference space.
 * @tparam ncomp Number of components in each degree of freedom.
 * @tparam etype Element type, i.e. Simplex = tris/tets, Tensor = quads/hexes
 * @tparam P Polynomial order if fixed, or use General if known only at runtime.
 */

//
// Element<T, dim, ncomp, etype, P>
//
template <int32 dim, int32 ncomp, ElemType etype, int32 P = Order::General>
class Element : public Element_impl<dim, ncomp, etype, P>
{
  protected:
  int32 m_el_id;

  public:
  using get_precision = Float;

  static constexpr int32 get_dim ()
  {
    return dim;
  }
  static constexpr uint32 get_ncomp ()
  {
    return ncomp;
  }
  static constexpr ElemType get_etype ()
  {
    return etype;
  }
  static constexpr int32 get_P ()
  {
    return P;
  }


  DRAY_EXEC static Element
  create (int32 el_id, SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p);
  DRAY_EXEC int32 get_el_id () const
  {
    return m_el_id;
  }
  DRAY_EXEC void construct (int32 el_id, SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p);
  DRAY_EXEC void construct (int32 el_id, SharedDofPtr<Vec<Float, ncomp>> dof_ptr);
  DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>> read_dof_ptr() const;
  DRAY_EXEC void get_bounds (AABB<ncomp> &aabb) const;
  DRAY_EXEC void get_sub_bounds (const SubRef<dim, etype> &sub_ref, AABB<ncomp> &aabb) const;

  DRAY_EXEC int32 get_order() const { return Element_impl<dim, ncomp, etype, P>::get_order(); }
};

//
// Element<T, dim, dim, etype, P>
//
template <int32 dim, ElemType etype, int32 P>
class Element<dim, dim, etype, P> : public InvertibleElement_impl<dim, etype, P>
{
  protected:
  int32 m_el_id;

  public:
  using get_precision = Float;
  static constexpr uint32 get_dim ()
  {
    return dim;
  }
  static constexpr uint32 get_ncomp ()
  {
    return dim;
  }
  static constexpr ElemType get_etype ()
  {
    return etype;
  }
  static constexpr int32 get_P ()
  {
    return P;
  }
  DRAY_EXEC static Element
  create (int32 el_id, SharedDofPtr<Vec<Float, dim>> dof_ptr, int32 p);
  DRAY_EXEC int32 get_el_id () const
  {
    return m_el_id;
  }
  DRAY_EXEC void construct (int32 el_id, SharedDofPtr<Vec<Float, dim>> dof_ptr, int32 p);
  DRAY_EXEC void construct (int32 el_id, SharedDofPtr<Vec<Float, dim>> dof_ptr);
  DRAY_EXEC SharedDofPtr<Vec<Float, dim>> read_dof_ptr() const;
  DRAY_EXEC void get_bounds (AABB<dim> &aabb) const;
  DRAY_EXEC void get_sub_bounds (const SubRef<dim, etype> &sub_ref, AABB<dim> &aabb) const;

  DRAY_EXEC int32 get_order() const { return Element_impl<dim, dim, etype, P>::get_order(); }
};


} // namespace dray


namespace dray
{

// TODO move sub_element_fixed_order() to pos_tensor_element.hpp

// sub_element_fixed_order()
template <uint32 RefDim, uint32 PhysDim, uint32 p_order, typename CoeffIterT = Vec<Float, PhysDim> *>
DRAY_EXEC MultiVec<Float, RefDim, PhysDim, p_order>
sub_element_fixed_order (const Range *ref_box, const CoeffIterT &coeff_iter);

} // namespace dray


// Implementations
namespace dray
{


//
// Element

// create()
template <int32 dim, int32 ncomp, ElemType etype, int32 P>
DRAY_EXEC Element<dim, ncomp, etype, P>
Element<dim, ncomp, etype, P>::create (int32 el_id, SharedDofPtr<Vec<Float, ncomp>> dof_ptr, int32 p)
{
  Element<dim, ncomp, etype, P> ret;
  ret.construct (el_id, dof_ptr, p);
  return ret;
}

// construct()
template <int32 dim, int32 ncomp, ElemType etype, int32 P>
DRAY_EXEC void
Element<dim, ncomp, etype, P>::construct (int32 el_id,
                                          SharedDofPtr<Vec<Float, ncomp>> dof_ptr,
                                          int32 p)
{
  Element_impl<dim, ncomp, etype, P>::construct (dof_ptr, p);
  m_el_id = el_id;
}

// construct()
template <int32 dim, int32 ncomp, ElemType etype, int32 P>
DRAY_EXEC void
Element<dim, ncomp, etype, P>::construct (int32 el_id, SharedDofPtr<Vec<Float, ncomp>> dof_ptr)
{
  Element_impl<dim, ncomp, etype, P>::construct (dof_ptr, -1);
  m_el_id = el_id;
}

// read_dof_ptr()
template <int32 dim, int32 ncomp, ElemType etype, int32 P>
DRAY_EXEC SharedDofPtr<Vec<Float, ncomp>>
Element<dim, ncomp, etype, P>::read_dof_ptr() const
{
  return Element_impl<dim, ncomp, etype, P>::read_dof_ptr();
}

// get_bounds()
template <int32 dim, int32 ncomp, ElemType etype, int32 P>
DRAY_EXEC void Element<dim, ncomp, etype, P>::get_bounds (AABB<ncomp> &aabb) const
{
  const int32 num_dofs = eattr::get_num_dofs( adapt_get_shape(*this),
                                              adapt_get_order_policy(*this, get_order()) );

  detail::positive_get_bounds<ncomp> (aabb, Element_impl<dim, ncomp, etype, P>::m_dof_ptr,
                                      num_dofs);
}

// get_sub_bounds()
template <int32 dim, int32 ncomp, ElemType etype, int32 P>
DRAY_EXEC void
Element<dim, ncomp, etype, P>::get_sub_bounds (const SubRef<dim, etype> &sub_ref,
                                               AABB<ncomp> &aabb) const
{
  Element_impl<dim, ncomp, etype, P>::get_sub_bounds (sub_ref, aabb);
}


//
// Element (nxn)

// create()
template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC Element<dim, dim, etype, P>
Element<dim, dim, etype, P>::create (int32 el_id, SharedDofPtr<Vec<Float, dim>> dof_ptr, int32 p)
{
  Element<dim, dim, etype, P> ret;
  ret.construct (el_id, dof_ptr, p);
  return ret;
}

// construct()
template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC void
Element<dim, dim, etype, P>::construct (int32 el_id, SharedDofPtr<Vec<Float, dim>> dof_ptr, int32 p)
{
  InvertibleElement_impl<dim, etype, P>::construct (dof_ptr, p);
  m_el_id = el_id;
}

// construct()
template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC void
Element<dim, dim, etype, P>::construct (int32 el_id, SharedDofPtr<Vec<Float, dim>> dof_ptr)
{
  InvertibleElement_impl<dim, etype, P>::construct (dof_ptr, -1);
  m_el_id = el_id;
}

// read_dof_ptr()
template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC SharedDofPtr<Vec<Float, dim>>
Element<dim, dim, etype, P>::read_dof_ptr() const
{
  return Element_impl<dim, dim, etype, P>::read_dof_ptr();
}

// get_bounds()
template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC void Element<dim, dim, etype, P>::get_bounds (AABB<dim> &aabb) const
{
  const int32 num_dofs = eattr::get_num_dofs( adapt_get_shape(*this),
                                              adapt_get_order_policy(*this, get_order()) );

  detail::positive_get_bounds<dim> (aabb, InvertibleElement_impl<dim, etype, P>::m_dof_ptr,
                                    num_dofs);
}

// get_sub_bounds()
template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC void
Element<dim, dim, etype, P>::get_sub_bounds (const SubRef<dim, etype> &sub_ref,
                                             AABB<dim> &aabb) const
{
  InvertibleElement_impl<dim, etype, P>::get_sub_bounds (sub_ref, aabb);
}


//
// InvertibleElement_impl

/// // TODO accept bounds on the solution.
/// template <int32 dim, ElemType etype, int32 P>
/// DRAY_EXEC bool
/// InvertibleElement_impl<dim, etype, P>::eval_inverse (const Vec<Float, dim> &world_coords,
///                                                      const SubRef<dim, etype> &guess_domain,
///                                                      Vec<Float, dim> &ref_coords,
///                                                      bool use_init_guess) const
/// {
///   stats::Stats stats; // dont need to construct because we never use this
///   // TODO: eliminate multiple versions of this function call
///   return eval_inverse (stats, world_coords, guess_domain, ref_coords, use_init_guess);
/// }


template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC bool
InvertibleElement_impl<dim, etype, P>::eval_inverse_local (const Vec<Float, dim> &world_coords,
                                                           Vec<Float, dim> &ref_coords) const
{
  stats::Stats stats; // dont need to construct because we never use this
  return eval_inverse_local (stats, world_coords, ref_coords);
}


template <int32 dim, ElemType etype, int32 P>
DRAY_EXEC bool
InvertibleElement_impl<dim, etype, P>::eval_inverse_local (stats::Stats &stats,
                                                           const Vec<Float, dim> &world_coords,
                                                           Vec<Float, dim> &ref_coords) const
{
  // Newton step to solve inverse of geometric transformation (assuming good initial guess).
  struct Stepper
  {
    DRAY_EXEC typename IterativeMethod::StepStatus operator() (Vec<Float, dim> &x) const
    {
      // project back onto the element
      for(int i = 0; i < dim; ++i)
      {
        x[i] = fminf(Float(1.f), fmaxf(x[i], Float(0.f)));
      }

      Vec<Float, dim> delta_y;
      Vec<Vec<Float, dim>, dim> j_col;
      Matrix<Float, dim, dim> jacobian;
      delta_y = m_transf.eval_d (x, j_col);
      delta_y = m_target - delta_y;

      for (int32 rdim = 0; rdim < dim; rdim++)
        jacobian.set_col (rdim, j_col[rdim]);

      bool inverse_valid;
      Vec<Float, dim> delta_x;
      delta_x = matrix_mult_inv (jacobian, delta_y, inverse_valid);

      //if (!inverse_valid) return IterativeMethod::Abort;

      x = x + delta_x;
      return IterativeMethod::Continue;
    }

    InvertibleElement_impl<dim, etype, P> m_transf;
    Vec<Float, dim> m_target;

  } stepper{ *this, world_coords };
  // TODO somewhere else in the program, figure out how to set the precision
  // based on the gradient and the image resolution.
  const Float tol_ref = 1e-4f;
  const int32 max_steps = 20;

  // Find solution.
  bool found = (IterativeMethod::solve (stats, stepper, ref_coords, max_steps,
                                        tol_ref) == IterativeMethod::Converged &&
                //this->is_inside (ref_coords));
                this->is_inside (ref_coords, tol_ref));
  return found;
}


/// template <int32 dim, ElemType etype, int32 P>
/// DRAY_EXEC bool
/// InvertibleElement_impl<dim, etype, P>::eval_inverse (stats::Stats &stats,
///                                                      const Vec<Float, dim> &world_coords,
///                                                      const SubRef<dim, etype> &guess_domain,
///                                                      Vec<Float, dim> &ref_coords,
///                                                      bool use_init_guess) const
/// {
///   using QueryT = Vec<Float, dim>;
///   using ElemT = InvertibleElement_impl<dim, etype, P>;
///   using RefBoxT = SubRef<dim, etype>;
///   using SolT = Vec<Float, dim>;
///
///   const Float tol_refbox = 1e-2f;
///   constexpr int32 subdiv_budget = 0;
///
///   RefBoxT domain = (use_init_guess ? guess_domain : ref_universe(RefSpaceTag<dim, etype>{}));
///
///   // For subdivision search, test whether the sub-element possibly contains the
///   // query point. Strict test because the bounding boxes are approximate.
///   struct FInBounds
///   {
///     DRAY_EXEC bool
///     operator() (stats::Stats &stats, const QueryT &query, const ElemT &elem, const RefBoxT &ref_box)
///     {
///       AABB<> bounds;
///       elem.get_sub_bounds (ref_box, bounds);
///       bool in_bounds = true;
///       for (int d = 0; d < dim; d++)
///         in_bounds = in_bounds && bounds.m_ranges[d].min () <= query[d] &&
///                     query[d] < bounds.m_ranges[d].max ();
///       return in_bounds;
///     }
///   };
///
///   // Get solution when close enough: Iterate using Newton's method.
///   struct FGetSolution
///   {
///     DRAY_EXEC bool operator() (stats::Stats &state,
///                                const QueryT &query,
///                                const ElemT &elem,
///                                const RefBoxT &ref_box,
///                                SolT &solution)
///     {
///       solution = ref_box.center (); // Awesome initial guess. TODO also use ref_box to guide the iteration.
///       return elem.eval_inverse_local (state, query, solution);
///     }
///   };
///
///   // Initiate subdivision search.
///   uint32 ret_code;
///   int32 num_solutions =
///   SubdivisionSearch::subdivision_search<QueryT, ElemT, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget> (
///   ret_code, stats, world_coords, *this, tol_refbox, &domain, &ref_coords, 1);
///
///   return num_solutions > 0;
/// }


// ------------

//
// sub_element_fixed_order()
//
template <uint32 RefDim, uint32 PhysDim, uint32 p_order, typename CoeffIterT>
DRAY_EXEC MultiVec<Float, RefDim, PhysDim, p_order>
sub_element_fixed_order (const Range *ref_box, const CoeffIterT &coeff_iter)
{
  using FixedBufferT = MultiVec<Float, RefDim, PhysDim, p_order>;

  // Copy coefficients from coeff_iter to coeff_buffer.
  FixedBufferT coeff_buffer;
  int32 ii = 0;
  for (auto &coeff : coeff_buffer.components ())
    coeff = coeff_iter[ii++];

  // Extract sub-patch along each dimension.
  SubPatch<RefDim, DeCasteljau>::template sub_patch_inplace<FixedBufferT, p_order> (
  coeff_buffer, ref_box, p_order);

  return coeff_buffer;
}

} // namespace dray

#include <dray/data_model/pos_simplex_element.tcc>
#include <dray/data_model/pos_tensor_element.tcc>

#endif // DRAY_ELEMENT_HPP
