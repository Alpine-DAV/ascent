// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SIMPLE_TENSOR_HPP
#define DRAY_SIMPLE_TENSOR_HPP

namespace dray
{

namespace detail
{

template <typename T> struct MultInPlace
{
  DRAY_EXEC static void mult (T *arr, T fac, int32 len)
  {
    for (int32 ii = 0; ii < len; ii++)
      arr[ii] *= fac;
  }
};

template <typename T, int32 S> struct MultInPlace<Vec<T, S>>
{
  DRAY_EXEC static void mult (Vec<T, S> *arr, Vec<T, S> fac, int32 len)
  {
    for (int32 ii = 0; ii < len; ii++)
      for (int32 c = 0; c < S; c++)
        arr[ii][c] *= fac[c];
  }
};

} //  namespace detail

template <typename T>
struct SimpleTensor // This means single product of some number of vectors.
{
  // Methods to compute a tensor from several vectors, in place.

  int32 s; // Be sure to initialize this before using.

  DRAY_EXEC int32 get_size_tensor (int32 t_order)
  {
    return pow (s, t_order);
  }

  // Where you should store the vectors that will
  // be used to construct the tensor.
  // First pointer is for the last/outermost looping
  // index variable in the tensor: X1(Y1Y2Y3)X2(Y1Y2Y3)X3(Y1Y2Y3).
  DRAY_EXEC void get_vec_init_ptrs (int32 t_order, T *arr, T **ptrs)
  {
    // We align all the initial vectors in the same direction,
    // along the innermost index. (That is, as blocks of contiguous memory.)
    // Each vector above the 0th must clear the orders below it.
    ptrs[0] = arr;
    for (int32 idx = 1, offset = s; idx < t_order; idx++, offset *= s)
      ptrs[idx] = arr + offset;
  }

  // After storing data in the vectors (at addresses returned by
  // get_vec_init_ptrs()), use this to construct the tensor product.
  //
  // Returns: The size of the tensor constructed.
  DRAY_EXEC int32 construct_in_place (int32 t_order, T *arr)
  {
    // This is a recursive method.
    if (t_order == 0)
    {
      return s;
    }
    else
    {
      int32 size_below = construct_in_place (t_order - 1, arr);
      // The current vector is safe because it was stored out of the way of lower construct().
      // Now The first 'size_below' addresses hold the sub-product of the lower vectors.
      // Construct the final tensor by multiplying the sub-product by each component of the current vector.
      // To do this in place, must overwrite the sub-product AFTER using it for the rest of the tensor.
      const T *cur_vec = arr + size_below;
      const T cur_head = cur_vec[0]; // Save this ahead of time.
      for (int32 layer = s - 1; layer >= 1; layer--)
      {
        const T comp = cur_vec[layer];
        memcpy (arr + layer * size_below, arr, size_below * sizeof (T));
        detail::MultInPlace<T>::mult (arr + layer * size_below, comp, size_below);
      }
      // Finish final layer by overwriting sub-product.
      detail::MultInPlace<T>::mult (arr, cur_head, size_below);
    }
  }
};

} // namespace dray

#endif
