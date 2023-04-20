// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/binomial.hpp>
#include <dray/math.hpp>

#include <algorithm>
#include <stdio.h> //DEBUG

namespace dray
{

bool BinomTable::size_at_least (int32 N) // N starts from 0.
{
  assert (N > 0);

  if (N > m_current_n)
  {
    const int32 new_n = std::max (N, 2 * m_current_n);
    const int32 new_size = (new_n + 1) * (new_n + 2) / 2;
    m_current_n = new_n;
    m_rows.resize (new_size);

    // Fill in all rows using Pascal's Triangle rule, and symmetry.
    int32 *row_ptr = m_rows.get_host_ptr ();
    for (int32 row_idx = 0; row_idx <= new_n; ++row_idx, row_ptr += row_idx)
    {
      // Row has (row_idx + 1) entries, 0 ... row_idx.

      // Fill ends with 1.
      row_ptr[0] = row_ptr[row_idx] = 1;

      // Fill middle using Pascal rule, working backward over the latter half of previous row.
      int32 *prev_ptr = row_ptr - 1;
      for (int32 k = 1; k <= row_idx / 2; k++, prev_ptr--)
      {
        row_ptr[k] = row_ptr[row_idx - k] = prev_ptr[0] + prev_ptr[-1];
      }
    }

    return true;
  }
  return false;
}


} // namespace dray
