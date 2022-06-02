// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_BINOMIAL_HPP
#define DRAY_BINOMIAL_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array.hpp>


#include <assert.h>

namespace dray
{

// Yet another alternative.
// This one stores the parameters used to generate.
class BinomRowIterator
{
  public:
  DRAY_EXEC void construct (int32 _n)
  {
    n = _n;
    k = 0;
    val = 1;
  }

  DRAY_EXEC void construct (int32 _n, int32 _k)
  {
    construct (_n);
    while (_k-- > 0)
      next ();
  }

  // Start at the beginning of the row.
  DRAY_EXEC void reset ()
  {
    k = 0;
    val = 1;
  }

  // Advance to next coefficient in the same row.
  DRAY_EXEC void next ()
  {
    val *= (n - k);
    k++;
    val /= k;
    if (k > n) reset ();
  }

  // Lower n but keep k fixed. Assumes that this is possible, i.e. 0 <= k <= n-1.
  DRAY_EXEC void lower_n ()
  {
    val *= (n - k);
    val /= n;
    n--;
  }

  DRAY_EXEC bool is_valid ()
  {
    return 0 <= k && k <= n;
  }

  // Getters.
  DRAY_EXEC int32 operator* ()
  {
    return val;
  }
  DRAY_EXEC int32 get_n ()
  {
    return n;
  }
  DRAY_EXEC int32 get_k ()
  {
    return k;
  }

  protected:
  int32 n;
  int32 k;
  int32 val;
};

// A table of binomial coefficients.
// Upon size_at_least() to more than the current row number,
// this will expand to twice the current row number.
// class BinomTable
struct BinomTable // DEBUG
{
  int32 m_current_n;
  Array<int32> m_rows;

  private:
  BinomTable ()
  {
    assert (false);
  };

  public:
  BinomTable (int32 N)
  {
    m_current_n = 0;
    size_at_least (N);
  }

  const int32 *get_host_ptr_const () const
  {
    return m_rows.get_host_ptr_const ();
  }
  const int32 *get_device_ptr_const () const
  {
    return m_rows.get_device_ptr_const ();
  }

  int32 get_current_n () const
  {
    return m_current_n;
  }

  // Returns whether resize took place.
  bool size_at_least (int32 N); // N starts from 0.

  DRAY_EXEC
  static int32 get (const int32 *row_ptr, int32 N, int32 k)
  {
    return get_row (row_ptr, N)[k];
  }

  DRAY_EXEC
  static const int32 *get_row (const int32 *row_ptr, int32 N)
  {
    return row_ptr + (N * (N + 1) / 2);
  }

  DRAY_EXEC
  static int32 *get_row (int32 *row_ptr, int32 N)
  {
    return row_ptr + (N * (N + 1) / 2);
  }
};

extern BinomTable GlobBinomTable;

template <typename T> struct BinomRow
{
  // Does not require any global state or precomputed values.
  DRAY_EXEC
  static void fill_single_row (int32 N, T *dest)
  {
    // Fill ends with 1.
    dest[0] = dest[N] = 1;

    // Fill middle entries sequentially based on entries to the left.
    // Use rule C(n,k) = (n-k+1)/k C(n,k-1)
    int32 prev = 1;
    for (int32 k = 1, nmkp1 = N; k <= N / 2; k++, nmkp1--)
    {
      // Integer division with less chance of overflow. (wikipedia.org)
      prev = (prev / k) * nmkp1 + (prev % k) * nmkp1 / k;
      dest[k] = dest[N - k] = static_cast<T> (prev);
    }
  }
};


//--- Template Meta Programming stuff ---//

// Recursive binomial coefficient template.
template <int32 n, int32 k> struct BinomT
{
  enum
  {
    val = BinomT<n - 1, k - 1>::val + BinomT<n - 1, k>::val
  };
};

// Base cases.
template <int32 n> struct BinomT<n, n>
{
  enum
  {
    val = 1
  };
};
template <int32 n> struct BinomT<n, 0>
{
  enum
  {
    val = 1
  };
};

namespace detail
{
// Hack: Inherited data members are layed out contiguously.
// Use inheritance to store constexpr Binom<> values one at a time recursively.
template <typename T, int32 n, int32 k>
struct BinomRowTInternal : public BinomRowTInternal<T, n, k - 1>
{
  const T cell = static_cast<T> (BinomT<n, k>::val);
};

// Base case.
template <typename T, int32 n> struct BinomRowTInternal<T, n, 0>
{
  const T cell = static_cast<T> (1);
};
} // namespace detail

// - To get a pointer to binomial coefficients stored in a static member:
//   const T *binomial_array = BinomRow<T,n>::get_static();
//
// - To store binomial coefficient literals in a local array variable,
//   1.    BinomRow<T,n> local_row;
//   2.    const T *local_binomial_array = local_row.get();
template <typename T, int32 n> class BinomRowT
{
  static detail::BinomRowTInternal<T, n, n> row_static; // Literals are fed into static member.
  detail::BinomRowTInternal<T, n, n> m_row; // Literals are fed into local memory.

  public:
  static const T *get_static ()
  {
    return (T *)&row_static;
  }

  DRAY_EXEC
  const T *get ()
  {
    return (T *)&m_row;
  }
};

template <typename T, int32 n>
detail::BinomRowTInternal<T, n, n> BinomRowT<T, n>::row_static;

} // namespace dray

#endif
