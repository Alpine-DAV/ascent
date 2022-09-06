// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SUBDIVISION_SEARCH_HPP
#define DRAY_SUBDIVISION_SEARCH_HPP

#include <dray/data_model/bernstein_basis.hpp>
#include <dray/utils/appstats.hpp>

#include <dray/aabb.hpp>

namespace dray
{

namespace detail
{
template <typename RefBox> struct SplitRefBox;
template <uint32 dim> struct SplitRefBox<AABB<dim>>;

template <typename RefBox> struct SplitRefBox
{
  // TODO need specializations for triangle/tetrahedra types.
  DRAY_EXEC static void
  split_ref_box (int32 depth, const RefBox &parent, RefBox &first_child, RefBox &second_child);
};


// TODO Shouldn't need full (explicit) specializations separately for 2D and 3D.
// However, without this I get undefined reference error. What is the right way
// to do this? Already tried instantiating classes in a .cpp file.
template <> struct SplitRefBox<AABB<2u>>
{
  DRAY_EXEC static void
  split_ref_box (int32 depth, const AABB<2u> &parent, AABB<2u> &first_child, AABB<2u> &second_child)
  {
    const int32 split_dim = parent.max_dim ();
    const float32 alpha = 0.5;

    first_child = parent;
    second_child = parent;
    parent.m_ranges[split_dim].split (alpha, first_child.m_ranges[split_dim],
                                      second_child.m_ranges[split_dim]);
  }
};

template <> struct SplitRefBox<AABB<3u>>
{
  DRAY_EXEC static void
  split_ref_box (int32 depth, const AABB<3u> &parent, AABB<3u> &first_child, AABB<3u> &second_child)
  {
    const int32 split_dim = parent.max_dim ();
    const float32 alpha = 0.5;

    first_child = parent;
    second_child = parent;
    parent.m_ranges[split_dim].split (alpha, first_child.m_ranges[split_dim],
                                      second_child.m_ranges[split_dim]);
  }
};

// TODO should be able to use a single implementation for both 2D and 3D.
/// template <uint32 dim>
/// struct SplitRefBox<AABB<dim>>
/// {
///   DRAY_EXEC static void split_ref_box(int32 depth, const AABB<dim> &parent, AABB<dim> &first_child, AABB<dim> &second_child)
///   {
///     const int32 split_dim = parent.max_dim();
///     const float32 alpha = 0.5;

///     first_child = parent;
///     second_child = parent;
///     parent.m_ranges[split_dim].split(alpha, first_child.m_ranges[split_dim], second_child.m_ranges[split_dim]);
///   }
/// };


template <typename RefBox>
DRAY_EXEC void
split_ref_box (int32 depth, const RefBox &parent, RefBox &first_child, RefBox &second_child)
{
  SplitRefBox<RefBox>::split_ref_box (depth, parent, first_child, second_child);
}


template <typename X>
DRAY_EXEC bool stack_push (X stack[], int32 &stack_sz, const int32 stack_cap, const X &x)
{
  bool ret;
  if ((ret = (stack_sz < stack_cap))) stack[stack_sz++] = x;
  return ret;
}

template <typename X>
DRAY_EXEC bool stack_pop (X stack[], int32 &stack_sz, X *&xp)
{
  bool ret;
  if ((ret = (stack_sz)))
    *xp = stack[--stack_sz];
  else
    xp = nullptr;
  return ret;
}
} // namespace detail

struct SubdivisionSearch
{
  enum ErrCode
  {
    NoError = 0u,
    MaxStack = 1u,
    MaxSubdiv = 2u
  };

  // Signatures
  //   DRAY_EXEC bool FInBounds::operator()(const Query &, const Elem &, const RefBox &);
  //   DRAY_EXEC bool FGetSolution::operator()(const Query &, const Elem &, const RefBox &, Sol &);

  template <typename Query, typename Elem, typename RefBox, typename Sol, typename FInBounds, typename FGetSolution, int32 subdiv_budget = 100, int32 stack_cap = 16>
  DRAY_EXEC static int32 subdivision_search (uint32 &ret_code,
                                             stats::Stats &stats,
                                             const Query &query,
                                             const Elem &elem,
                                             const Float ref_tol,
                                             RefBox *ref_box,
                                             Sol *solutions,
                                             const int32 list_cap = 1)
  {
    // Simulate a continuation-passing recursive version in which we would pass
    //   PointInCell(Element E, Stack<Element> stack, List<Solution> sols)
    // with the invariant that E may be nil only if stack is empty.
    //
    // In continuation-passing style, all state is re-packaged into the
    // arguments for the inner function call(s). When implemented as an
    // iterative procedure, the function arguments are modified in place.
    //
    // In general there could be multiple exit points, hence the while(true).
    //
    ret_code = 0;

    RefBox stack[stack_cap];
    int32 stack_sz = 0;
    int32 sol_sz = 0;

    int32 depth_stack[stack_cap];
    int32 depth = 0;

    int32 subdiv_balance = subdiv_budget;
    /// const int32 target_depth = 6;

    while (true)
    {
      if (ref_box != nullptr)
      {
        if (!subdiv_balance || ref_box->max_length () < ref_tol /*|| depth == target_depth*/) /* ref_box is 'leaf' */
        {
          if (!subdiv_balance) ret_code |= (uint32)MaxSubdiv;
          /* process leaf. i.e. decide whether to accept it. */
          Sol new_solution;
          bool is_solution = true;
          // std::cout<<"**************\n";
          is_solution = FGetSolution () (stats, query, elem, *ref_box, new_solution);
          // if(is_solution) std::cout<<"found\n";
          // else std::cout<<"NOT found\n";
          if (is_solution)
            detail::stack_push (solutions, sol_sz, list_cap, new_solution);
          if (sol_sz == list_cap) return sol_sz; // Early exit.

          if (stack_sz) depth = depth_stack[stack_sz - 1];
          detail::stack_pop (stack, stack_sz, ref_box); // return the continuation.
        } // is leaf

        else /* ref_box is not 'leaf'. */
        {
          depth++;
          subdiv_balance--;
          RefBox first, second;
          detail::split_ref_box (depth - 1, *ref_box, first, second); /// fprintf(stderr, "done splitting.\n");
          bool in_first = FInBounds () (stats, query, elem, first); /// fprintf(stderr, "got first bounds.\n");
          bool in_second = FInBounds () (stats, query, elem, second); /// fprintf(stderr, "got second bounds.\n");

          bool stack_full = false;
          if (in_first && in_second)
          {
            stack_full = !detail::stack_push (stack, stack_sz, stack_cap, second);
            if (stack_full)
              ret_code |= (uint32)MaxStack;
            else
              depth_stack[stack_sz - 1] = depth;
          }

          if (in_first)
            *ref_box = first; // return the continuation.
          else if (in_second)
            *ref_box = second; // return the continuation.
          else
          {
            if (stack_sz) depth = depth_stack[stack_sz - 1];
            detail::stack_pop (stack, stack_sz, ref_box); // return the continuation.
          }
        } // non leaf
      }
      else /*ref_box is nullptr*/
      {
        // Return solutions.
        return sol_sz;
      }
    }
  } // function subdivision_search()
};

namespace detail
{

/// template <uint32 dim>
/// DRAY_EXEC void
/// SplitRefBox<AABB<dim>>::split_ref_box( int32 depth,
///                                        const AABB<dim> &parent,
///                                        AABB<dim> &first_child,
///                                        AABB<dim> &second_child)
/// {
///   const int32 split_dim = parent.max_dim();
///   const float32 alpha = 0.5;

///   first_child = parent;
///   second_child = parent;
///   parent.m_ranges[split_dim].split(alpha, first_child.m_ranges[split_dim], second_child.m_ranges[split_dim]);
/// }


/// template <>
/// struct SplitRefBox<AABB<>>
/// {
///   DRAY_EXEC static void split_ref_box(int32 depth, const AABB<> &parent, AABB<> &first_child, AABB<> &second_child)
///   {
///     SplitRefBox<AABB<3>>::split_ref_box(depth, parent, first_child, second_child);
///   }
/// };
} // namespace detail

} // namespace dray

#endif // DRAY_SUBDIVISION_SEARCH_HPP
