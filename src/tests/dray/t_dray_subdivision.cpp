// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"

#include "dray/Element/bernstein_basis.hpp"
#include "dray/GridFunction/mesh.hpp"
#include "dray/aabb.hpp"
#include "dray/math.hpp"
#include "dray/range.hpp"
#include "dray/subdivision_search.hpp"

/*
 * Tests involving sub-elements and subdivision search.
 */


TEST (dray_subdivision, dray_subelement)
{
  /*TODO*/
}

template <int32 S> using RefBox = dray::AABB<S>;


TEST (dray_subdivision, dray_subdiv_search)
{
  constexpr dray::int32 p_order = 2;
  constexpr dray::int32 dim = 3;

  using Query = dray::Vec<dray::float32, dim>;
  using Elem = dray::Element<dim, dim, dray::ElemType::Quad, dray::Order::General>;
  using Sol = dray::Vec<dray::Float, dim>;
  using RefBox = RefBox<dim>;


  struct FInBounds
  {
    DRAY_EXEC bool
    operator() (dray::stats::Stats, const Query &query, const Elem &elem, const RefBox &ref_box)
    {
      /// fprintf(stderr, "FInBounds callback\n");
      dray::AABB<> bounds;
      elem.get_sub_bounds (ref_box, bounds);
      fprintf (stderr, "  aabb==[%.4f,%.4f,  %.4f,%.4f,  %.4f,%.4f]\n",
               bounds.m_ranges[0].min (), bounds.m_ranges[0].max (),
               bounds.m_ranges[1].min (), bounds.m_ranges[1].max (),
               bounds.m_ranges[2].min (), bounds.m_ranges[2].max ());
      return (
      bounds.m_ranges[0].min () <= query[0] && query[0] < bounds.m_ranges[0].max () &&
      bounds.m_ranges[1].min () <= query[1] && query[1] < bounds.m_ranges[1].max () &&
      bounds.m_ranges[2].min () <= query[2] && query[2] < bounds.m_ranges[2].max ());
    }
  };

  struct FGetSolution
  {
    DRAY_EXEC bool operator() (dray::stats::Stats,
                               const Query &query,
                               const Elem &elem,
                               const RefBox &ref_box,
                               Sol &solution)
    {
      /// fprintf(stderr, "FGetSolution callback\n");
      solution = ref_box.center (); // Awesome initial guess. TODO also use ref_box to guide the iteration.
      return elem.eval_inverse_local (query, solution);
    }
  };

  RefBox ref_box = RefBox::ref_universe ();
  Sol solution;

  Elem elem;
  constexpr dray::int32 num_dofs = dray::intPow (1 + p_order, dim);
  dray::Vec<dray::float32, dim> val_list[num_dofs] = // Identity map.
  {
    { 0.0, 0.0, 0.0 }, { 0.5, 0.0, 0.0 }, { 1.0, 0.0, 0.0 },

    { 0.0, 0.5, 0.0 }, { 0.5, 0.5, 0.0 }, { 1.0, 0.5, 0.0 },

    { 0.0, 1.0, 0.0 }, { 0.5, 1.0, 0.0 }, { 1.0, 1.0, 0.0 },


    { 0.0, 0.0, 0.5 }, { 0.5, 0.0, 0.5 }, { 1.0, 0.0, 0.5 },

    { 0.0, 0.5, 0.5 }, { 0.5, 0.5, 0.5 }, { 1.0, 0.5, 0.5 },

    { 0.0, 1.0, 0.5 }, { 0.5, 1.0, 0.5 }, { 1.0, 1.0, 0.5 },


    { 0.0, 0.0, 1.0 }, { 0.5, 0.0, 1.0 }, { 1.0, 0.0, 1.0 },

    { 0.0, 0.5, 1.0 }, { 0.5, 0.5, 1.0 }, { 1.0, 0.5, 1.0 },

    { 0.0, 1.0, 1.0 }, { 0.5, 1.0, 1.0 }, { 1.0, 1.0, 1.0 },
  };
  dray::int32 ctrl_idx_list[num_dofs] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 13, 14, 15, 16, 17,
                                          18, 19, 20, 21, 22, 23, 24, 25, 26 };

  dray::SharedDofPtr<dray::Vec<float32, dim>> dof_ptr{ ctrl_idx_list, val_list };
  elem.construct (0, dof_ptr, p_order);

  Query query = { 0.22, 0.33, 0.55 };

  /// {
  ///   dray::Vec<dray::float32, dim> result_val;
  ///   dray::Vec<dray::Vec<dray::float32, dim>, dim> result_deriv;
  ///   elem.eval({0.5, 0.5, 0.5}, result_val, result_deriv);
  ///   fprintf(stderr, "Elem eval: (%.4f, %.4f, %.4f)\n", result_val[0], result_val[1], result_val[2]);
  /// }

  const dray::float32 ref_tol = 1e-2;
  ;

  dray::uint32 ret_code;
  dray::stats::Stats stats;
  dray::int32 num_solutions =
  dray::SubdivisionSearch::subdivision_search<Query, Elem, RefBox, Sol, FInBounds, FGetSolution> (
  ret_code, stats, query, elem, ref_tol, &ref_box, &solution, 1);

  // Report results.
  fprintf (stderr, "Solution: (%f, %f, %f)\n", solution[0], solution[1], solution[2]);

  EXPECT_TRUE (num_solutions == 1);
  EXPECT_FLOAT_EQ (solution[0], query[0]);
  EXPECT_FLOAT_EQ (solution[1], query[1]);
  EXPECT_FLOAT_EQ (solution[2], query[2]);
}
