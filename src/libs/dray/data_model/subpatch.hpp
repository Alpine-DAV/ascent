// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SUBPATCH_HPP
#define DRAY_SUBPATCH_HPP

#include <dray/data_model/bernstein_basis.hpp>

namespace dray
{

// SplitDepth == how many axes, starting from outermost, should be split.
template <uint32 SplitDepth, typename Split1DMethod = DeCasteljau>
struct SubPatch;

// Base case.
template <typename Split1DMethod> struct SubPatch<1u, Split1DMethod>
{
  // Computes the Bernstein coefficients of a sub-patch by applying DeCasteljau twice.
  // If a non-negative argument to POrder is given,
  // that is used, else the argument to p_order is used.
  template <typename MultiArrayT, int32 POrder = -1>
  DRAY_EXEC static void
  sub_patch_inplace (MultiArrayT &elem_data, const Range *ref_box, uint32 p_order = 0)
  {
    const auto t1 = ref_box[0].max ();
    auto t0 = ref_box[0].min ();

    // Split left.
    if (t1 < 1.0)
      Split1DMethod::template split_inplace_left<MultiArrayT, POrder> (elem_data, t1, p_order);

    if (t1 > 0.0) t0 /= t1;

    // Split right.
    if (t0 > 0.0)
      Split1DMethod::template split_inplace_right<MultiArrayT, POrder> (elem_data, t0, p_order);
  }
};

// Arbitrary number of splits.
template <uint32 SplitDepth, typename Split1DMethod> struct SubPatch
{
  // Computes the Bernstein coefficients of a sub-patch by applying DeCasteljau twice per axis.
  // If a non-negative argument to POrder is given,
  // that is used, else the argument to p_order is used.
  template <typename MultiArrayT, int32 POrder = -1>
  DRAY_EXEC static void
  sub_patch_inplace (MultiArrayT &elem_data, const Range *ref_box, uint32 p_order = 0)
  {
    using ComponentT = typename FirstComponent<MultiArrayT, SplitDepth - 1>::component_t;

    const auto t1 = ref_box[0].max ();
    auto t0 = ref_box[0].min ();

    // Split left (outer axis).
    if (t1 < 1.0)
      for (auto &coeff_list : elem_data.template components<SplitDepth - 1> ())
        Split1DMethod::template split_inplace_left<ComponentT, POrder> (coeff_list, t1, p_order);

    if (t1 > 0.0) t0 /= t1;

    // Split right (outer axis).
    if (t0 > 0.0)
      for (auto &coeff_list : elem_data.template components<SplitDepth - 1> ())
        Split1DMethod::template split_inplace_right<ComponentT, POrder> (coeff_list, t0, p_order);

    // Split left/right (each inner axis).
    SubPatch<SplitDepth - 1, Split1DMethod>::template sub_patch_inplace<MultiArrayT, POrder> (
    elem_data, ref_box + 1, p_order);
  }
};

} // namespace dray

#endif // DRAY_SUBPATCH_HPP
