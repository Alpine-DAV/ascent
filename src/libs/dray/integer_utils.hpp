// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_INTEGER_UTILS_HPP
#define DRAY_INTEGER_UTILS_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/types.hpp>

namespace dray
{
template <int32 dim> class MultinomialCoeff;

using BinomialCoeff = MultinomialCoeff<1>;

template <int32 dim> class MultinomialCoeff
{
  // Invariant: m_val = MultinomialCoeff(n; i, j, k).
  // Invariant: i+j+k = n.
  protected:
  combo_int m_val;
  int32 m_n;
  int32 m_ijk[dim + 1];

  public:
  // Before using MultinomialCoeff, call construct(n).
  DRAY_EXEC void construct (int32 n)
  {
    constexpr int32 full_place = dim;
    m_val = 1;
    m_n = n;
    for (int32 d = 0; d <= dim; d++)
      m_ijk[d] = 0;
    m_ijk[full_place] = n;
  }

  // Getters.
  DRAY_EXEC combo_int get_val () const
  {
    return m_val;
  }
  DRAY_EXEC int32 get_n () const
  {
    return m_n;
  }
  DRAY_EXEC const int32 *get_ijk () const
  {
    return m_ijk;
  }

  // slice_over() - Advance to next coefficient along a direction.
  //                Be careful not to slide off Pascal's simplex.
  DRAY_EXEC combo_int slide_over (int32 inc_place)
  {
    constexpr int32 dec_place = dim;
    //       n!              n!         k
    // ---------------  =  -------  *  ---
    // (i+1)! M (k-1)!     i! M k!     i+1
    /// if (m_ijk[dec_place])
    int64 val = m_val;
    val *= m_ijk[dec_place];
    m_ijk[dec_place]--;

    m_ijk[inc_place]++;
    if (m_ijk[inc_place]) val /= m_ijk[inc_place];
    m_val = val;
    return m_val;
  }

  DRAY_EXEC combo_int slide_prev(int32 dec_place)
  {
    // Same as slide_over() but inc_place and dec_place are swapped.
    constexpr int32 inc_place = dim;
    int64 val = m_val;

    val *= m_ijk[dec_place];
    m_ijk[dec_place]--;
    m_ijk[inc_place]++;
    if (m_ijk[inc_place])
      val /= m_ijk[inc_place];

    m_val = val;
    return m_val;
  }

  // swap_places() - The multinomial coefficient is symmetric in i, j, k.
  DRAY_EXEC void swap_places (int32 place1, int32 place2 = dim)
  {
    const int32 s = m_ijk[place2];
    m_ijk[place2] = m_ijk[place1];
    m_ijk[place1] = s;
  }
};

} // namespace dray

#endif // DRAY_INTEGER_UTILS_HPP
