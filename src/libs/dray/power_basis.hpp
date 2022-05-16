// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_POWER_BASIS
#define DRAY_POWER_BASIS

namespace dray
{

//
// PowerBasis (Arbitrary dimension)
//
template <typename T, int32 RefDim>
struct PowerBasis : public PowerBasis<T, RefDim - 1>
{
  // -- Internals -- //

  int32 m_coeff_offset; // Set by init_shape().

  // -- Public -- //

  // Initializes p and coeff_offset, and returns offset.
  DRAY_EXEC int32 init_shape (int32 p)
  {
    return m_coeff_offset = (p + 1) * PowerBasis<T, RefDim - 1>::init_shape (p);
  }

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo (const Vec<T, RefDim> &xyz,
                               const CoeffIterType &coeff_iter,
                               Vec<T, PhysDim> &result_val,
                               Vec<Vec<T, PhysDim>, RefDim> &result_deriv) const;

  static constexpr int32 ref_dim = RefDim;
  int32 get_el_dofs () const
  {
    return (PowerBasis<T, 1>::p + 1) * m_coeff_offset;
  }
  int32 get_ref_dim () const
  {
    return RefDim;
  }

  static int32 get_aux_req ()
  {
    return 0;
  }
  static bool is_aux_req ()
  {
    return false;
  }

  // DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const;   //TODO

}; // PowerBasis (Arbitrary dimension)


//
// PowerBasis (1D - specialization for base case)
//
template <typename T> struct PowerBasis<T, 1>
{
  // -- Internals -- //
  int32 p; // Used by higher dimensions.

  // -- Public -- //

  // Returns offset of 1.
  DRAY_EXEC int32 init_shape (int32 _p)
  {
    p = _p;
    return 1;
  }

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC void linear_combo (const Vec<T, 1> &xyz,
                               const CoeffIterType &coeff_iter,
                               Vec<T, PhysDim> &ac_v,
                               Vec<Vec<T, PhysDim>, 1> &ac_dxyz) const
  {
    PowerBasis<T, 1>::linear_combo<CoeffIterType, PhysDim> (p, xyz[0], coeff_iter,
                                                            ac_v, ac_dxyz[0]);
  }

  static constexpr int32 ref_dim = 1;
  int32 get_el_dofs () const
  {
    return p + 1;
  }

  static int32 get_aux_req ()
  {
    return 0;
  }
  static bool is_aux_req ()
  {
    return false;
  }

  template <typename CoeffIterType, int32 PhysDim>
  DRAY_EXEC static void linear_combo (const int32 p,
                                      const T &x,
                                      const CoeffIterType &coeff_iter,
                                      Vec<T, PhysDim> &ac_v,
                                      Vec<T, PhysDim> &ac_dx)
  {
    ac_v = 0.0;
    ac_dx = 0.0;
    int32 k;
    for (k = p; k > 0; k--)
    {
      ac_v = ac_v * x + coeff_iter[k];
      ac_dx = ac_dx * x + coeff_iter[k] * k;
    }
    ac_v = ac_v * x + coeff_iter[k];
  }

  // DRAY_EXEC void calc_shape_dshape(const Vec<RefDim> &ref_pt, T *shape_val, Vec<RefDim> *shape_deriv) const;   //TODO

}; // PowerBasis 1D

//
// PowerBasis<T,RefDim>::linear_combo()
//
template <typename T, int32 RefDim>
template <typename CoeffIterType, int32 PhysDim>
DRAY_EXEC void
PowerBasis<T, RefDim>::linear_combo (const Vec<T, RefDim> &xyz,
                                     const CoeffIterType &coeff_iter,
                                     Vec<T, PhysDim> &ac_v,
                                     Vec<Vec<T, PhysDim>, RefDim> &ac_dxyz) const
{
  // Local so compiler can figure it out.
  const int32 &p = PowerBasis<T, 1>::p;

  // Local const so we don't modify ourself.
  const int32 coeff_offset = m_coeff_offset;

  // Initialize all accumulators to zero.
  ac_v = 0.0;
  for (int32 r = 0; r < RefDim; r++)
    ac_dxyz[r] = 0.0;

  // Aliases to separate x-component from yz-components.
  const T &x = xyz[0];
  const Vec<T, RefDim - 1> &yz = *((const Vec<T, RefDim - 1> *)&xyz[1]);

  Vec<T, PhysDim> &ac_dx = ac_dxyz[0];
  Vec<Vec<T, PhysDim>, RefDim - 1> &ac_dyz =
  *((Vec<Vec<T, PhysDim>, RefDim - 1> *)&ac_dxyz[1]);

  // Variables to hold results of "inner" summations.
  Vec<T, PhysDim> ac_v_i;
  Vec<Vec<T, PhysDim>, RefDim - 1> ac_dyz_i; // Note dx is absent from inner.

  int32 k;
  for (k = p; k > 0; k--)
  {
    PowerBasis<T, RefDim - 1>::linear_combo (yz, coeff_iter + k * coeff_offset,
                                             ac_v_i, ac_dyz_i);
    ac_v = ac_v * x + ac_v_i;
    for (int32 r = 0; r < RefDim - 1; r++)
      ac_dyz[r] = ac_dyz[r] * x + ac_dyz_i[r];
    ac_dx = ac_dx * x + ac_v_i * k;
  }
  PowerBasis<T, RefDim - 1>::linear_combo (yz, coeff_iter + k * coeff_offset, ac_v_i, ac_dyz_i);
  ac_v = ac_v * x + ac_v_i;
  for (int32 r = 0; r < RefDim - 1; r++)
    ac_dyz[r] = ac_dyz[r] * x + ac_dyz_i[r];
}
} // namespace dray

#endif
