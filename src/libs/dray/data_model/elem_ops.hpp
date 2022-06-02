// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ELEM_OPS_HPP
#define DRAY_ELEM_OPS_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/element.hpp>
#include <dray/data_model/dof_access.hpp>

namespace dray
{

  namespace detail {
    constexpr int32 cartesian_to_tri_idx(int32 i, int32 j, int32 elen)
    {
      // i runs fastest, j slowest.
      // There are a total of (elen)(elen+1)/2 vertices in the triangle.
      // (idx - i) counts the number of vertices below the cap, so
      //
      //   (elen)(elen+1)/2 - (idx - i) = (elen-j)(elen-j+1)/2
      //
      //   j(1 + 2*elen - j)/2 + i = idx

      return (2*elen + 1 - j)*j/2 + i;
    }

    constexpr int32 cartesian_to_tet_idx(int32 i, int32 j, int32 k, int32 e)
    {
      // i runs fastest, k slowest.
      // There are a total of (elen)(elen+1)(elen+2)/6 vertices in the tetrahedron.
      // (idx - cartesian_to_tri_idx(i,j,elen-k)) counts
      // the number of vertices below the cap, so
      //
      //   (elen)(elen+1)(elen+2)/6 - (idx - (2*elen + 1 - j)*j/2 - i)
      //   = (elen-k)(elen-k+1)(elen-k+2)/6
      //
      //   (e)(e+1)(e+2)/6 - (e-k)(e+1-k)(e+2-k)/6 + (2e + 1 - j)*j/2 + i = idx
      //
      //   ((k - 3e - 3)(k) + (3e + 6)e + 2)k/6 + (2e + 1 - j)*j/2 + i = idx

      return (((-1-e)*3+k)*k + (3*e + 6)*e + 2)*k/6 + (2*e + 1 - j)*j/2 + i;
    }
  }


  /// // get_sub_bounds<Simplex>
  /// template <int32 dim, int32 ncomp, int32 P>
  /// DRAY_EXEC void get_sub_bounds(
  ///     const ShapeTAG,                            // Tag for shape
  ///     const OrderPolicy<P> order_p,              // Tag/data for order policy
  ///     WriteDofPtr<Vec<Float, ncomp>> &dof_ptr,                 // dofs read and written here
  ///     const Split<ElemType::Simplex> &split)
  /// {
  ///   //TODO split the triangle element and use coefficients from split element.
  ///   //For now it just uses the non-split coefficients
  ///   //and returns bounds for entire element.

  ///   const int num_dofs = get_num_dofs(ShapeTAG{}, order_p);

  ///
  /// }

  namespace eops
  {
    /** QuadEdgeWalker */
    template <int32 P>
    struct QuadEdgeWalker
    {
      DRAY_EXEC constexpr QuadEdgeWalker(const OrderPolicy<P> order_p, const int32 eid)
        : m_order_p(order_p),
          m_p(eattr::get_order(order_p)),
          m_base( (m_p+1)*(m_p * quad_props::quad_eoffset1(eid))
                         +(m_p * quad_props::quad_eoffset0(eid)) ),
          m_di( quad_props::quad_estride(eid, m_p+1) )
      {}

      DRAY_EXEC int32 edge2quad(int32 i) const { return m_base + i * m_di; }

      const OrderPolicy<P> m_order_p;
      const int32 m_p;
      const int32 m_base;
      const int32 m_di;
    };

    /** TriEdgeWalker */
    template <int32 P>
    struct TriEdgeWalker
    {
      DRAY_EXEC constexpr TriEdgeWalker(const OrderPolicy<P> order_p, const int32 eid)
        : m_order_p(order_p),
          m_p(eattr::get_order(order_p)),
          m_base( tri_props::tri_eoffset(eid) * m_p ),
          m_di( tri_props::tri_estep(eid) )
      {}

      DRAY_EXEC int32 edge2tri(int32 i) const
      {
        const Vec<uint8, 2> cart = m_base + m_di * i;
        return ::dray::detail::cartesian_to_tri_idx(cart[0], cart[1], m_p+1);
      }

      const OrderPolicy<P> m_order_p;
      const int32 m_p;
      const Vec<uint8, 2> m_base;
      const Vec<uint8, 2> m_di;
    };

    /** HexEdgeWalker */
    template <int32 P>
    struct HexEdgeWalker
    {
      DRAY_EXEC constexpr HexEdgeWalker(const OrderPolicy<P> order_p, const int32 eid)
        : m_order_p(order_p),
          m_p(eattr::get_order(order_p)),
          m_base( (m_p+1)*(m_p+1)*(m_p*hex_props::hex_eoffset2(eid))
                             + (m_p+1)*(m_p*hex_props::hex_eoffset1(eid))
                                   + 1*(m_p*hex_props::hex_eoffset0(eid)) ),
          m_di( hex_props::hex_estride(eid, m_p+1) )
      {}

      DRAY_EXEC int32 edge2hex(int32 i) const { return m_base + i * m_di; }

      const OrderPolicy<P> m_order_p;
      const int32 m_p;
      const int32 m_base;
      const int32 m_di;
    };

    /** HexFaceWalker */
    template <int32 P>
    struct HexFaceWalker
    {
      DRAY_EXEC constexpr HexFaceWalker(const OrderPolicy<P> order_p, const int32 fid)
        : m_order_p(order_p),
          m_p(eattr::get_order(order_p)),
          m_base( (m_p+1)*(m_p+1)*(m_p*hex_props::hex_foffset2(fid))
                           +(m_p+1)*(m_p*hex_props::hex_foffset1(fid))
                                   +(m_p*hex_props::hex_foffset0(fid)) ),
          m_di( hex_props::hex_fstrideU(fid, m_p+1) ),
          m_dj( hex_props::hex_fstrideV(fid, m_p+1) )
      {}

      DRAY_EXEC int32 face2hex(int32 i, int32 j) const { return m_base + j * m_dj + i * m_di; }

      const OrderPolicy<P> m_order_p;
      const int32 m_p;
      const int32 m_base;
      const int32 m_di;
      const int32 m_dj;
    };


    /** eval_d_edge(ShapeHex, Linear) */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d_edge( ShapeHex,
                                             const OrderPolicy<Linear> order_p,
                                             const int32 eid,
                                             const ReadDofPtr<Vec<Float, ncomp>> &C,
                                             const Vec<Float, 1> &rc,
                                             Vec<Vec<Float, ncomp>, 1> &out_deriv )
    {
      constexpr int32 p = eattr::get_order(order_p.as_cxp());
      HexEdgeWalker<Linear> hew(order_p.as_cxp(), eid);
      const Vec<Float, ncomp> C0 = C[ hew.edge2hex(0) ];
      const Vec<Float, ncomp> C1 = C[ hew.edge2hex(1) ];
      out_deriv[0] = (C1 - C0)*p;
      return (C1 * rc[0]) + (C0 * (1-rc[0]));
    }

    /** eval_d_edge(ShapeHex, Quadratic) */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d_edge( ShapeHex,
                                             const OrderPolicy<Quadratic> order_p,
                                             const int32 eid,
                                             const ReadDofPtr<Vec<Float, ncomp>> &C,
                                             const Vec<Float, 1> &rc,
                                             Vec<Vec<Float, ncomp>, 1> &out_deriv )
    {
      constexpr int32 p = eattr::get_order(order_p.as_cxp());
      const HexEdgeWalker<Quadratic> hew(order_p.as_cxp(), eid);
      Vec<Float, ncomp> C0 = C[ hew.edge2hex(0) ];
      Vec<Float, ncomp> C1 = C[ hew.edge2hex(1) ];
      Vec<Float, ncomp> C2 = C[ hew.edge2hex(2) ];

      C0 = (C1 * rc[0]) + (C0 * (1-rc[0]));
      C1 = (C2 * rc[0]) + (C1 * (1-rc[0]));

      out_deriv[0] = (C1 - C0)*p;
      return (C1 * rc[0]) + (C0 * (1-rc[0]));
    }

    struct BinomialCoeffTable
    {
      // TODO specify gpu 'constant memory' for binomial coefficients.
      combo_int m_table[MaxPolyOrder+1];

      DRAY_EXEC BinomialCoeffTable(int32 p)
      {
        BinomialCoeff binomial_coeff;
        binomial_coeff.construct(p);
        for (int32 ii = 0; ii <= p; ii++)
        {
          m_table[ii] = binomial_coeff.get_val();
          binomial_coeff.slide_over(0);
        }
      }

      DRAY_EXEC const combo_int & operator[](int32 i) const { return m_table[i]; }
    };

    /** eval_d_edge(ShapeHex, General) */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d_edge( ShapeHex,
                                             const OrderPolicy<General> order_p,
                                             const int32 eid,
                                             const ReadDofPtr<Vec<Float, ncomp>> &C,
                                             const Vec<Float, 1> &rc,
                                             Vec<Vec<Float, ncomp>, 1> &out_deriv )
    {
      const int32 p = eattr::get_order(order_p);
      const HexEdgeWalker<General> hew(order_p, eid);
      const Float &u = rc[0], _u = 1.0-u;
      Float upow = 1.0;

      out_deriv = 0;

      if (p == 0)
        return C[0];

      BinomialCoeffTable B(p);
      Vec<Float, ncomp> result;
      result = 0;

      for (int32 i = 0; i <= p; ++i)
      {
        Vec<Float, ncomp> Ci = C[hew.edge2hex(i)] * B[i];

        result *= _u;
        if (i > 0)
        {
          out_deriv[0] += Ci * (i*upow);
          upow *= u;
        }
        if (i < p)
        {
          out_deriv[0] *= _u;
          out_deriv[0] += Ci * (-(p-i)*upow);
        }
        result += Ci * upow;
      }

      return result;
    }


    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_1d(const OrderPolicy<General> order_p,
                                        const Vec<Float, ncomp> *C,
                                        const Float &u,
                                        const Float &len)
    {
      const int32 p = eattr::get_order(order_p);
      const Float _u = len-u;
      Float upow = 1.0;

      if (p == 0)
        return C[0];

      BinomialCoeffTable B(p);
      Vec<Float, ncomp> result;
      result = 0;

      for (int32 i = 0; i <= p; ++i)
      {
        Vec<Float, ncomp> Ci = C[i] * B[i];
        result *= _u;
        if (i > 0)
          upow *= u;
        result += Ci * upow;
      }

      return result;
    }


    /** eval_d_face(ShapeHex, Linear) */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d_face( ShapeHex,
                                             const OrderPolicy<Linear> order_p,
                                             const int32 fid,
                                             const ReadDofPtr<Vec<Float, ncomp>> &C,
                                             const Vec<Float, 2> &rc,
                                             Vec<Vec<Float, ncomp>, 2> &out_deriv )
    {
      constexpr int32 p = eattr::get_order(order_p.as_cxp());
      const HexFaceWalker<Linear> hfw(order_p.as_cxp(), fid);

      const Float &u = rc[0],  _u = 1.0-u;
      const Float &v = rc[1],  _v = 1.0-v;

      const Vec<Float, ncomp> C00 = C[hfw.face2hex(0,0)];
      const Vec<Float, ncomp> C01 = C[hfw.face2hex(1,0)];
      const Vec<Float, ncomp> C10 = C[hfw.face2hex(0,1)];
      const Vec<Float, ncomp> C11 = C[hfw.face2hex(1,1)];

      out_deriv[0] = ((C11-C10)*p)*v + ((C01-C00)*p)*_v;
      out_deriv[1] = ((C11-C01)*p)*u + ((C10-C00)*p)*_u;
      return (C11*u + C10*_u)*v + (C01*u + C00*_u)*_v;
    }

    /** eval_d_face(ShapeHex, Quadratic) */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d_face( ShapeHex,
                                             const OrderPolicy<Quadratic> order_p,
                                             const int32 fid,
                                             const ReadDofPtr<Vec<Float, ncomp>> &C,
                                             const Vec<Float, 2> &rc,
                                             Vec<Vec<Float, ncomp>, 2> &out_deriv )
    {
      constexpr int32 p = eattr::get_order(order_p.as_cxp());
      const HexFaceWalker<Quadratic> hfw(order_p.as_cxp(), fid);

      const Float &u = rc[0],  _u = 1.0-u;
      const Float &v = rc[1],  _v = 1.0-v;

      Vec<Float, ncomp> C00 = C[hfw.face2hex(0, 0)];
      Vec<Float, ncomp> C01 = C[hfw.face2hex(1, 0)];
      Vec<Float, ncomp> C02 = C[hfw.face2hex(2, 0)];
      C00 = C01*u + C00*_u;  //DeCasteljau
      C01 = C02*u + C01*_u;  //DeCasteljau

      Vec<Float, ncomp> C10 = C[hfw.face2hex(0, 1)];
      Vec<Float, ncomp> C11 = C[hfw.face2hex(1, 1)];
      Vec<Float, ncomp> C12 = C[hfw.face2hex(2, 1)];
      C10 = C11*u + C10*_u;  //DeCasteljau
      C11 = C12*u + C11*_u;  //DeCasteljau

      Vec<Float, ncomp> C20 = C[hfw.face2hex(0, 2)];
      Vec<Float, ncomp> C21 = C[hfw.face2hex(1, 2)];
      Vec<Float, ncomp> C22 = C[hfw.face2hex(2, 2)];
      C20 = C21*u + C20*_u;  //DeCasteljau
      C21 = C22*u + C21*_u;  //DeCasteljau

      C00 = C10*v + C00*_v;  //DeCasteljau
      C10 = C20*v + C10*_v;  //DeCasteljau

      C01 = C11*v + C01*_v;  //DeCasteljau
      C11 = C21*v + C11*_v;  //DeCasteljau

      out_deriv[0] = ((C11-C10)*p)*v + ((C01-C00)*p)*_v;
      out_deriv[1] = ((C11-C01)*p)*u + ((C10-C00)*p)*_u;
      return (C11*u + C10*_u)*v + (C01*u + C00*_u)*_v;
    }


    /** eval_d_face(ShapeHex, General) */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d_face( ShapeHex,
                                             const OrderPolicy<General> order_p,
                                             const int32 fid,
                                             const ReadDofPtr<Vec<Float, ncomp>> &C,
                                             const Vec<Float, 2> &rc,
                                             Vec<Vec<Float, ncomp>, 2> &out_deriv )
    {
      const int32 p = eattr::get_order(order_p);
      const HexFaceWalker<General> hfw(order_p, fid);

      const Float &u = rc[0],  _u = 1.0-u;
      const Float &v = rc[1],  _v = 1.0-v;

      out_deriv = 0;
      if (p == 0)
        return C[0];

      BinomialCoeffTable B(p);

      Vec<Float, ncomp> result_uv, result_u, dr_u;
      result_uv = 0;
      Float vpow = 1.0;

      for (int32 j = 0; j <= p; ++j)
      {
        result_u = 0;
        dr_u = 0;
        Float upow = 1.0;
        for (int32 i = 0; i <= p; ++i)
        {
          Vec<Float, ncomp> Ci = C[hfw.face2hex(i,j)];
          Ci *= B[i];

          result_u *= _u;
          if (i > 0)
          {
            dr_u += Ci * (i*upow);
            upow *= u;
          }
          if (i < p)
          {
            dr_u *= _u;
            dr_u += Ci * (-(p-i)*upow);
          }
          result_u += Ci * upow;
        }
        result_u *= B[j];
        dr_u *= B[j];

        result_uv *= _v;
        out_deriv[0] *= _v;
        if (j > 0)
        {
          out_deriv[1] += result_u * (j*vpow);
          vpow *= v;
        }
        if (j < p)
        {
          out_deriv[1] *= _v;
          out_deriv[1] += result_u * (-(p-j)*vpow);
        }
        result_uv += result_u * vpow;
        out_deriv[0] += dr_u * vpow;

      }

      return result_uv;
    }


    /* eval_d() */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeHex,
                                        const OrderPolicy<Constant> order_p,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 3> &rc,
                                        Vec<Vec<Float, ncomp>, 3> &out_deriv )
    {
      for (int32 i = 0; i < 3; i++)
        for (int32 c = 0; c < ncomp; c++)
          out_deriv[i][c] = 0;
      return C[0];
    }

    /* eval_d() */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeHex,
                                        const OrderPolicy<Linear> order_p,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 3> &rc,
                                        Vec<Vec<Float, ncomp>, 3> &out_deriv )
    {
      const Float &u = rc[0],  _u = 1.0-u;
      const Float &v = rc[1],  _v = 1.0-v;
      const Float &w = rc[2],  _w = 1.0-w;

      out_deriv[0] = (C[1] - C[0]) * (_v) * (_w) +
                     (C[3] - C[2]) * v * (_w) +
                     (C[5] - C[4]) * (_v) * w +
                     (C[7] - C[6]) * v * w;

      out_deriv[1] = (C[2] - C[0]) * (_u) * (_w) +
                     (C[3] - C[1]) * u * (_w) +
                     (C[6] - C[4]) * (_u) * w +
                     (C[7] - C[5]) * u * w;

      out_deriv[2] = (C[4] - C[0]) * (_u) * (_v) +
                     (C[5] - C[1]) * u * (_v) +
                     (C[6] - C[2]) * (_u) * v +
                     (C[7] - C[3]) * u * v;

      return  C[0] * ((_u) * (_v) * (_w))
            + C[1] * (u * (_v) * (_w))
            + C[2] * ((_u) * v * (_w))
            + C[3] * (u * v * (_w))
            + C[4] * ((_u) * (_v) * w)
            + C[5] * (u * (_v) * w)
            + C[6] * ((_u) * v * w)
            + C[7] * (u * v * w);
    }


    /* eval_d() */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeHex,
                                        const OrderPolicy<Quadratic> order_p,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 3> &r,
                                        Vec<Vec<Float, ncomp>, 3> &out_deriv )
    {
      // Shape functions. Quadratic has 3 1D shape functions on each axis.
      const Float su[3] = { (1 - r[0]) * (1 - r[0]), 2 * r[0] * (1 - r[0]), r[0] * r[0] };
      const Float sv[3] = { (1 - r[1]) * (1 - r[1]), 2 * r[1] * (1 - r[1]), r[1] * r[1] };
      const Float sw[3] = { (1 - r[2]) * (1 - r[2]), 2 * r[2] * (1 - r[2]), r[2] * r[2] };

      // Shape derivatives.
      const Float dsu[3] = { -2*(1-r[0]), 2 - 4*r[0], 2*r[0] };
      const Float dsv[3] = { -2*(1-r[1]), 2 - 4*r[1], 2*r[1] };
      const Float dsw[3] = { -2*(1-r[2]), 2 - 4*r[2], 2*r[2] };

      out_deriv[0] =
          C[0] * dsu[0] * sv[0] * sw[0] +
          C[1] * dsu[1] * sv[0] * sw[0] + C[2] * dsu[2] * sv[0] * sw[0] +
          C[3] * dsu[0] * sv[1] * sw[0] + C[4] * dsu[1] * sv[1] * sw[0] +
          C[5] * dsu[2] * sv[1] * sw[0] + C[6] * dsu[0] * sv[2] * sw[0] +
          C[7] * dsu[1] * sv[2] * sw[0] + C[8] * dsu[2] * sv[2] * sw[0] +

          C[9] * dsu[0] * sv[0] * sw[1] + C[10] * dsu[1] * sv[0] * sw[1] +
          C[11] * dsu[2] * sv[0] * sw[1] +
          C[12] * dsu[0] * sv[1] * sw[1] + C[13] * dsu[1] * sv[1] * sw[1] +
          C[14] * dsu[2] * sv[1] * sw[1] + C[15] * dsu[0] * sv[2] * sw[1] +
          C[16] * dsu[1] * sv[2] * sw[1] + C[17] * dsu[2] * sv[2] * sw[1] +

          C[18] * dsu[0] * sv[0] * sw[2] +
          C[19] * dsu[1] * sv[0] * sw[2] + C[20] * dsu[2] * sv[0] * sw[2] +
          C[21] * dsu[0] * sv[1] * sw[2] + C[22] * dsu[1] * sv[1] * sw[2] +
          C[23] * dsu[2] * sv[1] * sw[2] + C[24] * dsu[0] * sv[2] * sw[2] +
          C[25] * dsu[1] * sv[2] * sw[2] + C[26] * dsu[2] * sv[2] * sw[2];

      out_deriv[1] =
          C[0] * su[0] * dsv[0] * sw[0] +
          C[1] * su[1] * dsv[0] * sw[0] + C[2] * su[2] * dsv[0] * sw[0] +
          C[3] * su[0] * dsv[1] * sw[0] + C[4] * su[1] * dsv[1] * sw[0] +
          C[5] * su[2] * dsv[1] * sw[0] + C[6] * su[0] * dsv[2] * sw[0] +
          C[7] * su[1] * dsv[2] * sw[0] + C[8] * su[2] * dsv[2] * sw[0] +

          C[9] * su[0] * dsv[0] * sw[1] + C[10] * su[1] * dsv[0] * sw[1] +
          C[11] * su[2] * dsv[0] * sw[1] +
          C[12] * su[0] * dsv[1] * sw[1] + C[13] * su[1] * dsv[1] * sw[1] +
          C[14] * su[2] * dsv[1] * sw[1] + C[15] * su[0] * dsv[2] * sw[1] +
          C[16] * su[1] * dsv[2] * sw[1] + C[17] * su[2] * dsv[2] * sw[1] +

          C[18] * su[0] * dsv[0] * sw[2] +
          C[19] * su[1] * dsv[0] * sw[2] + C[20] * su[2] * dsv[0] * sw[2] +
          C[21] * su[0] * dsv[1] * sw[2] + C[22] * su[1] * dsv[1] * sw[2] +
          C[23] * su[2] * dsv[1] * sw[2] + C[24] * su[0] * dsv[2] * sw[2] +
          C[25] * su[1] * dsv[2] * sw[2] + C[26] * su[2] * dsv[2] * sw[2];

      out_deriv[2] =
          C[0] * su[0] * sv[0] * dsw[0] +
          C[1] * su[1] * sv[0] * dsw[0] + C[2] * su[2] * sv[0] * dsw[0] +
          C[3] * su[0] * sv[1] * dsw[0] + C[4] * su[1] * sv[1] * dsw[0] +
          C[5] * su[2] * sv[1] * dsw[0] + C[6] * su[0] * sv[2] * dsw[0] +
          C[7] * su[1] * sv[2] * dsw[0] + C[8] * su[2] * sv[2] * dsw[0] +

          C[9] * su[0] * sv[0] * dsw[1] + C[10] * su[1] * sv[0] * dsw[1] +
          C[11] * su[2] * sv[0] * dsw[1] +
          C[12] * su[0] * sv[1] * dsw[1] + C[13] * su[1] * sv[1] * dsw[1] +
          C[14] * su[2] * sv[1] * dsw[1] + C[15] * su[0] * sv[2] * dsw[1] +
          C[16] * su[1] * sv[2] * dsw[1] + C[17] * su[2] * sv[2] * dsw[1] +

          C[18] * su[0] * sv[0] * dsw[2] +
          C[19] * su[1] * sv[0] * dsw[2] + C[20] * su[2] * sv[0] * dsw[2] +
          C[21] * su[0] * sv[1] * dsw[2] + C[22] * su[1] * sv[1] * dsw[2] +
          C[23] * su[2] * sv[1] * dsw[2] + C[24] * su[0] * sv[2] * dsw[2] +
          C[25] * su[1] * sv[2] * dsw[2] + C[26] * su[2] * sv[2] * dsw[2];

      return C[0] * su[0] * sv[0] * sw[0] +
             C[1] * su[1] * sv[0] * sw[0] + C[2] * su[2] * sv[0] * sw[0] +
             C[3] * su[0] * sv[1] * sw[0] + C[4] * su[1] * sv[1] * sw[0] +
             C[5] * su[2] * sv[1] * sw[0] + C[6] * su[0] * sv[2] * sw[0] +
             C[7] * su[1] * sv[2] * sw[0] + C[8] * su[2] * sv[2] * sw[0] +

             C[9] * su[0] * sv[0] * sw[1] + C[10] * su[1] * sv[0] * sw[1] +
             C[11] * su[2] * sv[0] * sw[1] +
             C[12] * su[0] * sv[1] * sw[1] +
             C[13] * su[1] * sv[1] * sw[1] +
             C[14] * su[2] * sv[1] * sw[1] +
             C[15] * su[0] * sv[2] * sw[1] +
             C[16] * su[1] * sv[2] * sw[1] +
             C[17] * su[2] * sv[2] * sw[1] +

             C[18] * su[0] * sv[0] * sw[2] +
             C[19] * su[1] * sv[0] * sw[2] +
             C[20] * su[2] * sv[0] * sw[2] +
             C[21] * su[0] * sv[1] * sw[2] +
             C[22] * su[1] * sv[1] * sw[2] +
             C[23] * su[2] * sv[1] * sw[2] +
             C[24] * su[0] * sv[2] * sw[2] +
             C[25] * su[1] * sv[2] * sw[2] +
             C[26] * su[2] * sv[2] * sw[2];
    }


    /* eval_d() */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeHex,
                                        const OrderPolicy<General> order_p,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 3> &rc,
                                        Vec<Vec<Float, ncomp>, 3> &out_deriv )
    {
      const int32 p = eattr::get_order(order_p);
      const Float &u = rc[0], _u = 1.0-u;
      const Float &v = rc[1], _v = 1.0-v;
      const Float &w = rc[2], _w = 1.0-w;

      out_deriv = 0;
      if (p == 0)
        return C[0];

      Vec<Float, ncomp> C_all;
      C_all = 0;

      Vec<Float, ncomp> &Du_all = out_deriv[0];
      Vec<Float, ncomp> &Dv_all = out_deriv[1];
      Vec<Float, ncomp> &Dw_all = out_deriv[2];

      BinomialCoeffTable B(p);

      // Level 3: Reduce over k --> into C_all, Du_all, Dv_all, Dw_all
      Float wpow = 1.0;
      for (int32 k = 0; k <= p; ++k)
      {
        Vec<Float, ncomp> Ck, Duk, Dvk;
        Ck = 0;
        Duk = 0;
        Dvk = 0;

        // Level 2: Reduce over j --> into Ck, Duk, Dvk
        Float vpow = 1.0;
        for (int32 j = 0; j <= p; ++j)
        {
          Vec<Float, ncomp> Cj, Duj;
          Cj = 0;
          Duj = 0;

          // Level 1: Reduce over i --> into Cj, Duj
          Float upow = 1.0;
          for (int32 i = 0; i <= p; ++i)
          {
            // Level 0: Read --> into Ci.
            Vec<Float, ncomp> Ci = C[i + j*(p+1) + k*(p+1)*(p+1)] * B[i];

            // Fold Ci into value and derivatives.
            Cj *= _u;
            if (i > 0)
            {
              Duj += Ci * (i*upow);
              upow *= u;
            }
            if (i < p)
            {
              Duj *= _u;
              Duj += Ci * (-(p-i)*upow);
            }
            Cj += Ci * upow;
          }
          Cj *= B[j];
          Duj *= B[j];

          // Fold Cj, Duj into value and derivatives.
          Ck *= _v;
          Duk *= _v;
          if (j > 0)
          {
            Dvk += Cj * (j*vpow);
            vpow *= v;
          }
          if (j < p)
          {
            Dvk *= _v;
            Dvk += Cj * (-(p-j)*vpow);
          }
          Ck += Cj * vpow;
          Duk += Duj * vpow;
        }
        Ck *= B[k];
        Duk *= B[k];
        Dvk *= B[k];

        // Fold Ck, Duk, Dvk into value and derivatives.
        C_all *= _w;
        Du_all *= _w;
        Dv_all *= _w;
        if (k > 0)
        {
          Dw_all += Ck * (k*wpow);
          wpow *= w;
        }
        if (k < p)
        {
          Dw_all *= _w;
          Dw_all += Ck * (-(p-k)*wpow);
        }
        C_all += Ck * wpow;
        Du_all += Duk * wpow;
        Dv_all += Dvk * wpow;
      }

      return C_all;
    }




    /** eval_d() */
    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTri,
                                        OrderPolicy<Linear>,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 2> &rc,
                                        Vec<Vec<Float, ncomp>, 2> &out_deriv )
    {
      // C[2]
      // C[0] C[1]
      const Float &u = rc[0], &v = rc[1];
      const Float t = 1.0f - u - v;

      Float sd[3];
      sd[0] = -1.0f;

      // du
      sd[1] = 1.0f;
      sd[2] = 0.0f;
      // leaving this for clarity but will remove multiplies by zero
      //out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];
      out_deriv[0] = C[0] * sd[0] + C[1] * sd[1];

      // dv
      sd[1] = 0.0f;
      sd[2] = 1.0f;
      // leaving this for clarity but will remove multiplies by zero
      //out_deriv[1] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];
      out_deriv[1] = C[0] * sd[0] + C[2] * sd[2];

      const Float s[3] = { t, u, v };
      return C[0] * s[0] + C[1] * s[1] + C[2] * s[2];
    }

    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTri,
                                        OrderPolicy<Quadratic>,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 2> &rc,
                                        Vec<Vec<Float, ncomp>, 2> &out_deriv )
    {
      // C[6]
      //
      // C[3] C[4]
      //
      // C[0] C[1] C[2]
      const Float &u = rc[0], &v = rc[1];
      const Float t = 1.0f - u - v;

      Float sd[6];
      sd[0] = 2*(-t);

      // -------------------------------
      // du
                        sd[1] = 2*(t-u);    sd[2] = 2*u;
      sd[3] = 2*(-v);   sd[4] = 2*(v);
      sd[5] = 0.0f;
      //
      out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2]
                   + C[3] * sd[3] + C[4] * sd[4]
                   + C[5] * sd[5];
      // -------------------------------

      // -------------------------------
      // dv
                         sd[1] = 2*(-u);   sd[2] = 0.0f;
      sd[3] = 2*(t-v);   sd[4] = 2*(u);
      sd[5] = 2*v;
      //
      out_deriv[1] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2]
                   + C[3] * sd[3] + C[4] * sd[4]
                   + C[5] * sd[5];
      // -------------------------------


      const Float s[6] = { t*t,     2*t*u,   u*u,
                           2*t*v,   2*v*u,
                           v*v };
      return C[0] * s[0] + C[1] * s[1] + C[2] * s[2] +
             C[3] * s[3] + C[4] * s[4] +
             C[5] * s[5];
    }



    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTet,
                                        OrderPolicy<Linear>,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 3> &rc,
                                        Vec<Vec<Float, ncomp>, 3> &out_deriv )
    {
      //  C[2]
      //
      //  C[0]  C[1]
      // C[3]
      const Float &u = rc[0], &v = rc[1], &w = rc[2];
      const Float t = 1.0f - u - v - w;

      Float sd[4];
      sd[0] = -1.0f;

      // du
      sd[1] = 1.0f;   sd[2] = 0.0f;   sd[3] = 0.0f;
      out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

      // du
      sd[1] = 0.0f;   sd[2] = 1.0f;   sd[3] = 0.0f;
      out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

      // dw
      sd[1] = 0.0f;   sd[2] = 0.0f;   sd[3] = 1.0f;
      out_deriv[0] = C[0] * sd[0] + C[1] * sd[1] + C[2] * sd[2];

      const Float s[4] = { t, u, v, w };
      return C[0] * s[0] + C[1] * s[1] + C[2] * s[2] + C[3] * s[3];
    }



    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTet,
                                        OrderPolicy<Quadratic>,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 3> &rc,
                                        Vec<Vec<Float, ncomp>, 3> &out_deriv )
    {
      //  Behold, the P2 tetrahedron
      //
      //              v=1
      //
      //              C[5]
      //             /|   `
      //            / C[3]  C[4]
      //           C[8]        `
      //          /|  C[0]--C[1]--C[2]   u=1
      //         / C[6]   C[7]  '
      //        C[9]   '
      //    w=1
      //
      const Float &u = rc[0], &v = rc[1], &w = rc[2];
      const Float t = 1.0f - u - v - w;

      Float sd[10];
      sd[0] = 2*(-t);

      // -------------------------------
      // du
                        sd[1] = 2*(t-u);   sd[2] = 2*u;
      sd[3] = 2*(-v);   sd[4] = 2*(v);
      sd[5] = 0.0f;
                          sd[6] = 2*(-w);    sd[7] = 2*(w);
                          sd[8] = 0.0f;
                                                sd[9] = 0.0f;
      out_deriv[0] = 0;
      for (int32  i = 0; i < 10; ++i)
        out_deriv[0] += C[i] * sd[i];
      // -------------------------------

      // -------------------------------
      // dv
                         sd[1] = 2*(-u);  sd[2] = 0.0f;
      sd[3] = 2*(t-v);   sd[4] = 2*(u);
      sd[5] = 2*v;
                          sd[6] = 2*(-w);    sd[7] = 0.0f;
                          sd[8] = 2*(w);
                                                sd[9] = 0.0f;
      out_deriv[1] = 0;
      for (int32  i = 0; i < 10; ++i)
        out_deriv[1] += C[i] * sd[i];
      // -------------------------------

      // -------------------------------
      // dv

                       sd[1] = 2*(-u);   sd[2] = 0.0f;
      sd[3] = 2*(-v);  sd[4] = 0.0f;
      sd[5] = 0.0f;
                         sd[6] = 2*(t-w);   sd[7] = 2*(u);
                         sd[8] = 2*(v);
                                              sd[9] = 2*w;
      out_deriv[2] = 0;
      for (int32  i = 0; i < 10; ++i)
        out_deriv[2] += C[i] * sd[i];
      // -------------------------------


      const Float s[10] = { t*t,     2*t*u,   u*u,
                            2*t*v,   2*v*u,
                            v*v,
                                       2*t*w,  2*u*w,
                                       2*v*w,
                                                  w*w };

      Vec<Float, ncomp> ret;  ret = 0;
      for (int32 i = 0; i < 10; ++i)
        ret += C[i] * s[i];
      return ret;
    }


    template <int32 ncomp>
    DRAY_EXEC Vec<Float, ncomp> eval_d( ShapeTet,
                                        OrderPolicy<General> order_p,
                                        const ReadDofPtr<Vec<Float, ncomp>> &C,
                                        const Vec<Float, 3> &rc,
                                        Vec<Vec<Float, ncomp>, 3> &out_deriv )
    {
      // Implementation migrated from Element (pos_simplex_element.tcc)

      using DofT = Vec<Float, ncomp>;
      using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

      const int32 p = eattr::get_order(order_p);
      const int32 pm1 = p-1;

      if (p == 0)
      {
        out_deriv[0] = 0.0;
        out_deriv[1] = 0.0;
        out_deriv[2] = 0.0;
        return C[0];
      }

      // The dof offset in an axis depends on the index in that axis and lesser axes.
      // The offset can be derived from the linearization formula.
      // Note: D^d(p) is the number of dofs in a d-dimensional p-order simplex, or nchoosek(p+d,d).
      //
      //     l(i,j,k) =   \sum_{k'=0}^[k-1} D^2(p-k')
      //                + \sum_{j'=0}^{j-1} D^1(p-k-j')
      //                + \sum_{i'=0}^{i-1} D^0(p-k-j-i')
      //
      //              = ...
      //
      //              =   D^3(p)     - D^3(p-k)
      //                + D^2(p-k)   - D^2(p-k-j)
      //                + D^1(p-k-j) - D^1(p-k-j-i)
      //
      //     \delta l^0 (i,j,k) = D^0(p-k-j-i)  = 1
      //
      //     \delta l^1 (i,j,k) = D^1(p-k-j) - D^0(p-k-j) + D^0(p-k-j-i)
      //                        = D^1(p-k-j) - D^0(p-k-j) + \delta l^0 (i,j,k)  = p-k-j+1
      //
      //     \delta l^2 (i,j,k) = D^2(p-k) - D^1(p-k) + D^1(p-k-j) - D^0(p-k-j) + D^0(p-k-j-i)
      //                        = D^2(p-k) - D^1(p-k) + \delta l^1(i,j,k)   = (p-k+1)(p-k+2)/2 - j

      PtrT dof_ptr = C; // Make a local copy that can be incremented.

      // Barycentric coordinates.
      const Float &u = rc[0];
      const Float &v = rc[1];
      const Float &w = rc[2];
      const Float t = Float (1.0) - (u + v + w);

      // Multinomial coefficient. Will traverse Pascal's simplex using
      // transitions between adjacent multinomial coefficients (slide_over()),
      // and transpositions back to the start of each row (swap_places()).
      MultinomialCoeff<3> mck;
      mck.construct (pm1);

      int32 dof_idx = 0;

      DofT k_sum;
      k_sum = 0.0;
      Vec<DofT, 3u> k_sum_d;
      k_sum_d = 0.0;
      Float wpow = 1.0;
      for (int32 kk = 0; kk <= pm1; kk++)
      {
        const int32 sz_p_k = (p - kk + 1) * (p - kk + 2) / (1 * 2); // nchoosek(p-kk + dim-1, dim-1)

        DofT j_sum;
        j_sum = 0.0;
        Vec<DofT, 3u> j_sum_d;
        j_sum_d = 0.0;
        Float vpow = 1.0;
        for (int32 jj = 0; jj <= (pm1 - kk); jj++)
        {
          const int32 sz_p_j = (p - kk - jj + 1) / 1; // nchoosek(q-jj + dim-2, dim-2)

          DofT i_sum;
          i_sum = 0.0;
          Vec<DofT, 3u> i_sum_d;
          i_sum_d = 0.0;
          Float upow = 1.0;
          for (int32 ii = 0; ii <= (pm1 - kk - jj); ii++)
          {
            // Horner's rule innermost, due to decreasing powers of t (mu = pm1 - kk - jj - ii).
            i_sum *= t;
            i_sum_d[0] *= t;
            i_sum_d[1] *= t;
            i_sum_d[2] *= t;

            const DofT dof_mu = dof_ptr[dof_idx];
            const Vec<DofT, 3u> dof_ijk = { dof_ptr[dof_idx + 1], // Offset dofs
                                            dof_ptr[dof_idx + sz_p_j],
                                            dof_ptr[dof_idx + sz_p_k - jj] };
            dof_idx++;

            i_sum += (dof_mu * t + dof_ijk[0] * u + dof_ijk[1] * v + dof_ijk[2] * w) *
                     (mck.get_val () * upow);
            i_sum_d[0] += (dof_ijk[0] - dof_mu) * (mck.get_val () * upow);
            i_sum_d[1] += (dof_ijk[1] - dof_mu) * (mck.get_val () * upow);
            i_sum_d[2] += (dof_ijk[2] - dof_mu) * (mck.get_val () * upow);

            upow *= u;
            if (ii < (pm1 - kk - jj)) mck.slide_over (0);
          }
          mck.swap_places (0);

          dof_idx++; // Skip end of row.

          j_sum += i_sum * vpow;
          j_sum_d += i_sum_d * vpow;
          vpow *= v;
          if (jj < (pm1 - kk)) mck.slide_over (1);
        }
        mck.swap_places (1);

        dof_idx++; // Skip tip of triangle.

        k_sum += j_sum * wpow;
        k_sum_d += j_sum_d * wpow;
        wpow *= w;
        if (kk < pm1) mck.slide_over (2);
      }
      // mck.swap_places(2);

      out_deriv = k_sum_d * p;
      return k_sum;
    }




  /**
   * RotatedIdx
   *
   *   @brief An multi-index + a permutation for local ordering + linearizer.
   *          Purpose: iterate over faces/edges of a 3D element when all threads do same thing.
   *
   *   ApparentCoefficient[i,j] = dofs[ oriented_idx.linearize(i,j) ]
   *                            == dofs[ m_linearizer( {indices with i applied to axis pi[0],
   *                                                                 j applied to axis pi[1],...}) ];
   */
  // RotatedIdx3, for hex
  template <int8 pi0, int8 pi1, int8 pi2, typename LinearizerT>
  struct RotatedIdx3
  {
    protected:
      mutable Vec<int32, 3> m_I;
      const LinearizerT m_linearizer;

      DRAY_EXEC void apply(int32 i)                   const { m_I[pi0]+=i; }
      DRAY_EXEC void apply(int32 i, int32 j)          const { m_I[pi0]+=i; m_I[pi1]+=j; }
      DRAY_EXEC void apply(int32 i, int32 j, int32 k) const { m_I[pi0]+=i; m_I[pi1]+=j; m_I[pi2]+=k; }

    public:
      DRAY_EXEC RotatedIdx3(int32 start0, int32 start1, int32 start2, const LinearizerT & linearizer)
      : m_I{{start0, start1, start2}},
        m_linearizer(linearizer)
      { }

      DRAY_EXEC int32 linearize(int32 i) const
      {
        apply(i);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2]);
        apply(-i);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j) const
      {
        apply(i, j);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2]);
        apply(-i, -j);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j, int32 k) const
      {
        apply(i, j, k);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2]);
        apply(-i, -j, -k);
        return idx;
      }
  };

  // RotatedIdx4, for tet
  template <int8 pi0, int8 pi1, int8 pi2, int8 pi3, typename LinearizerT>
  struct RotatedIdx4
  {
    protected:
      mutable Vec<int32, 4> m_I;
      const LinearizerT m_linearizer;

      DRAY_EXEC void apply(int32 i)                            const { m_I[pi0]+=i; }
      DRAY_EXEC void apply(int32 i, int32 j)                   const { m_I[pi0]+=i; m_I[pi1]+=j; }
      DRAY_EXEC void apply(int32 i, int32 j, int32 k)          const { m_I[pi0]+=i; m_I[pi1]+=j; m_I[pi2]+=k; }
      DRAY_EXEC void apply(int32 i, int32 j, int32 k, int32 l) const { m_I[pi0]+=i; m_I[pi1]+=j; m_I[pi2]+=k; m_I[pi3]+=l; }

    public:
      DRAY_EXEC RotatedIdx4(int32 start0, int32 start1, int32 start2, int32 start3, const LinearizerT & linearizer)
      : m_I{{start0, start1, start2, start3}},
        m_linearizer(linearizer)
      { }

      DRAY_EXEC int32 linearize(int32 i) const
      {
        apply(i);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j) const
      {
        apply(i, j);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i, -j);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j, int32 k) const
      {
        apply(i, j, k);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i, -j, -k);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j, int32 k, int32 l) const
      {
        apply(i, j, k, l);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i, -j, -k, -l);
        return idx;
      }
  };


  struct HexFlat
  {
    int32 m_order;
    DRAY_EXEC int32 operator()(int32 i, int32 j, int32 k) const
    {
      return i + (m_order+1)*j + (m_order+1)*(m_order+1)*k;
    }
  };

  struct TetFlat
  {
    int32 m_order;
    DRAY_EXEC int32 operator()(int32 i, int32 j, int32 k, int32 mu) const
    {
      return ::dray::detail::cartesian_to_tet_idx(i, j, k, (m_order+1));
    }
  };















  }//eops



  // --------------------------------------------------------------------------
  // split_inplace()
  // --------------------------------------------------------------------------

  namespace detail
  {
    constexpr int32 cartesian_to_tri_idx(int32 i, int32 j, int32 edge);
    constexpr int32 cartesian_to_tet_idx(int32 i, int32 j, int32 k, int32 e);
  }

  // The Split<> object describes a binary split of the simplex at some point
  // (given by 'factor') along an edge (given by vtx_displaced and vtx_tradeoff).
  // Each row of coefficients parallel to the specified edge undergoes
  // 1D-DeCasteljau subdivision. The side closest to the 'tradeoff' vertex is
  // treated as the fixed, exterior node. The side closest to the 'displaced'
  // vertex is treated as the parameter-dependent, interior node.
  //
  //              .                 .           .           .
  //             .-*               . .         . .         . .
  //            .-*-*             . .-*       . . .       . . .
  //           .-*-*-*           . .-*-*     . . .-*     . . . .
  //       (v1=p)    (v0=p)
  //     tradeoff    displaced
  //
  // Subject to axis permutations, the splitting can be carried out as below:
  //
  //   // Triangle
  //   for (0 <= h < p)
  //     for (p-h >= num_updates >= 1)
  //       for (p-h >= v0 > p-h - num_updates, v0+v1 = p-h)
  //         C[v0,v1;h] := f*C[v0,v1;h] + (1-f)*C[v0-1,v1+1;h];
  //
  //   // Tetrahedron
  //   for (0 <= g+h < p)
  //     for (p-(g+h) >= num_updates >= 1)
  //       for (p-(g+h) >= v0 > p-(g+h) - num_updates, v0+v1 = p-(g+h))
  //         C[v0,v1;g,h] := f*C[v0,v1;g,h] + (1-f)*C[v0-1,v1+1;g,h];
  //

  //
  // split_inplace<2, Simplex>        (Triangle)
  //
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      ShapeTri, OrderPolicy<P> order_p,
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Simplex> &split)
  {
    const uint8 p = (uint8) eattr::get_order(order_p);
    const uint8 v0 = (uint8) split.vtx_displaced;
    const uint8 v1 = (uint8) split.vtx_tradeoff;
    const uint8 v2 = 0+1+2 - v0 - v1;

    uint8 b[3];  // barycentric indexing

    // I think this way of expressing the permuation is most readable.
    // On the other hand, potential for loop unrolling might be easier to
    // detect if the permutation was expressed using the inverse.

    for (b[v2] = 0; b[v2] < p; ++b[v2])
      for (uint8 num_updates = p-b[v2]; num_updates >= 1; --num_updates)
        for (b[v0] = p-b[v2]; b[v0] > p-b[v2] - num_updates; --b[v0])
        {
          b[v1] = p-b[v2]-b[v0];

          uint8 b_left[3];
          b_left[v0] = b[v0] - 1;
          b_left[v1] = b[v1] + 1;
          b_left[v2] = b[v2];

          const uint32 right = detail::cartesian_to_tri_idx(b[0], b[1], p+1);
          const uint32 left = detail::cartesian_to_tri_idx(b_left[0], b_left[1], p+1);

          dof_ptr[right] =
              dof_ptr[right] * split.factor + dof_ptr[left] * (1-split.factor);
        }
  }

  /** @deprecated */
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<2, ncomp, ElemType::Simplex, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Simplex> &split)
  {
    split_inplace(ShapeTri{},
                  eattr::adapt_create_order_policy(OrderPolicy<P>{}, elem_info.get_order()),
                  dof_ptr,
                  split);
  }



  //
  // split_inplace<3, Simplex>          (Tetrahedron)
  //
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      ShapeTet, OrderPolicy<P> order_p,
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Simplex> &split)
  {
    const uint8 p = (uint8) eattr::get_order(order_p);
    const uint8 v0 = (uint8) split.vtx_displaced;
    const uint8 v1 = (uint8) split.vtx_tradeoff;

    const uint8 avail = -1u & ~(1u << v0) & ~(1u << v1);
    const uint8 v2 = (avail & 1u) ? 0 : (avail & 2u) ? 1 : (avail & 4u) ? 2 : 3;
    const uint8 v3 = 0+1+2+3 - v0 - v1 - v2;

    uint8 b[4];  // barycentric indexing

    for (b[v3] = 0; b[v3] < p; ++b[v3])
      for (b[v2] = 0; b[v2] < p-b[v3]; ++b[v2])
      {
        const uint8 gph = b[v2] + b[v3];

        for (uint8 num_updates = p-gph; num_updates >= 1; --num_updates)
          for (b[v0] = p-gph; b[v0] > p-gph - num_updates; --b[v0])
          {
            b[v1] = p-gph-b[v0];

            int8 b_left[4];
            b_left[v0] = b[v0] - 1;
            b_left[v1] = b[v1] + 1;
            b_left[v2] = b[v2];
            b_left[v3] = b[v3];

            const uint32 right = detail::cartesian_to_tet_idx(
                b[0], b[1], b[2], p+1);
            const uint32 left = detail::cartesian_to_tet_idx(
                b_left[0], b_left[1], b_left[2], p+1);

            dof_ptr[right] =
                dof_ptr[right] * split.factor + dof_ptr[left] * (1-split.factor);
          }
      }
  }

  /** @deprecated */
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<3, ncomp, ElemType::Simplex, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Simplex> &split)
  {
    split_inplace(ShapeTet{},
                  eattr::adapt_create_order_policy(OrderPolicy<P>{}, elem_info.get_order()),
                  dof_ptr,
                  split);
  }

  // Binary split on quad:
  //
  //  left:
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //     .-*-*-*    . .-*-*    . . .-*    . . . .
  //
  //  right:
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //     *-*-*-.    *-*-. .    *-. . .    . . . .
  //

  //
  // split_inplace<Tensor>
  //
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      ShapeHex, OrderPolicy<P> order_p,
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Tensor> &split)
  {
    constexpr int32 dim = eattr::get_dim(ShapeHex{});
    const uint32 p = eattr::get_order(order_p);

    uint32 p_order_pow[4];
    p_order_pow[0] = 1;
    p_order_pow[1] = p_order_pow[0] * (p + 1);
    p_order_pow[2] = p_order_pow[1] * (p + 1);
    p_order_pow[3] = p_order_pow[2] * (p + 1);

    const int32 &axis = split.axis;
    static_assert((1 <= dim && dim <= 3), "split_inplace() only supports 1D, 2D, or 3D.");
    assert((0 <= axis && axis < dim));
    const uint32 stride = p_order_pow[axis];
    const uint32 chunk_sz = p_order_pow[axis+1];
    const uint32 num_chunks = p_order_pow[dim - (axis+1)];

    if (!split.f_lower_t_upper)
    {
      // Left
      for (int32 chunk = 0; chunk < num_chunks; ++chunk, dof_ptr += chunk_sz)
      {
        // Split the chunk along axis.
        // If there are axes below axis, treat them as a vector of dofs.

        // In DeCasteljau left split, we repeatedly overwrite the right side.
        for (int32 from_front = 1; from_front <= p; ++from_front)
          for (int32 ii = p; ii >= 0+from_front; --ii)
            for (int32 e = 0; e < stride; ++e)
            {
              dof_ptr[ii*stride + e] =
                  dof_ptr[(ii-1)*stride + e] * (1 - split.factor)
                  + dof_ptr[ii*stride + e] * (split.factor);
            }
      }
    }
    else
    {
      // Right
      for (int32 chunk = 0; chunk < num_chunks; ++chunk, dof_ptr += chunk_sz)
      {
        // Split the chunk along axis.
        // If there are axes below axis, treat them as a vector of dofs.

        // In DeCasteljau right split, we repeatedly overwrite the left side.
        for (int32 from_back = 1; from_back <= p; ++from_back)
          for (int32 ii = 0; ii <= p-from_back; ++ii)
            for (int32 e = 0; e < stride; ++e)
            {
              dof_ptr[ii*stride + e] =
                  dof_ptr[ii*stride + e] * (1 - split.factor)
                  + dof_ptr[(ii+1)*stride + e] * (split.factor);
            }
      }
    }
  }

  //
  // split_inplace<Tensor>
  //
  template <int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      ShapeQuad, OrderPolicy<P> order_p,
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Tensor> &split)
  {
    constexpr int32 dim = eattr::get_dim(ShapeQuad{});
    const uint32 p = eattr::get_order(order_p);

    uint32 p_order_pow[4];
    p_order_pow[0] = 1;
    p_order_pow[1] = p_order_pow[0] * (p + 1);
    p_order_pow[2] = p_order_pow[1] * (p + 1);
    p_order_pow[3] = p_order_pow[2] * (p + 1);

    const int32 &axis = split.axis;
    static_assert((1 <= dim && dim <= 3), "split_inplace() only supports 1D, 2D, or 3D.");
    assert((0 <= axis && axis < dim));
    const uint32 stride = p_order_pow[axis];
    const uint32 chunk_sz = p_order_pow[axis+1];
    const uint32 num_chunks = p_order_pow[dim - (axis+1)];

    if (!split.f_lower_t_upper)
    {
      // Left
      for (int32 chunk = 0; chunk < num_chunks; ++chunk, dof_ptr += chunk_sz)
      {
        // Split the chunk along axis.
        // If there are axes below axis, treat them as a vector of dofs.

        // In DeCasteljau left split, we repeatedly overwrite the right side.
        for (int32 from_front = 1; from_front <= p; ++from_front)
          for (int32 ii = p; ii >= 0+from_front; --ii)
            for (int32 e = 0; e < stride; ++e)
            {
              dof_ptr[ii*stride + e] =
                  dof_ptr[(ii-1)*stride + e] * (1 - split.factor)
                  + dof_ptr[ii*stride + e] * (split.factor);
            }
      }
    }
    else
    {
      // Right
      for (int32 chunk = 0; chunk < num_chunks; ++chunk, dof_ptr += chunk_sz)
      {
        // Split the chunk along axis.
        // If there are axes below axis, treat them as a vector of dofs.

        // In DeCasteljau right split, we repeatedly overwrite the left side.
        for (int32 from_back = 1; from_back <= p; ++from_back)
          for (int32 ii = 0; ii <= p-from_back; ++ii)
            for (int32 e = 0; e < stride; ++e)
            {
              dof_ptr[ii*stride + e] =
                  dof_ptr[ii*stride + e] * (1 - split.factor)
                  + dof_ptr[(ii+1)*stride + e] * (split.factor);
            }
      }
    }
  }

  /** @deprecated */
  template <int32 dim, int32 ncomp, int32 P>
  DRAY_EXEC void split_inplace(
      const Element<dim, ncomp, ElemType::Tensor, P> &elem_info,  // tag for template + order
      WriteDofPtr<Vec<Float, ncomp>> dof_ptr,
      const Split<ElemType::Tensor> &split)
  {
    split_inplace(Shape<dim, Tensor>{},
                  eattr::adapt_create_order_policy(OrderPolicy<P>{}, elem_info.get_order()),
                  dof_ptr,
                  split);

  }

  // --------------------------------------------------------------------------



  // Imported from isosurface_meshing

/// namespace DeCasteljauSplitting
/// {
/// // sub_element()
/// //
/// // Replaces sub_element_fixed_order()
/// // This version operates in-place and does not assume fixed order.
/// template <uint32 dim, uint32 ncomp>
/// DRAY_EXEC void sub_element(uint32 p_order,
///                            const Range *ref_boxs,
///                            WriteDofPtr<Vec<Float, ncomp>> &wptr)
/// {
///   // Split along each axis sequentially. It is a tensor element.
///   for (int32 d = 0; d < dim; ++d)
///   {
///     const Float t1 = ref_boxs[d].max();
///     Float t0 = ref_boxs[d].min();
///
///     // Split left, using right endpoint.
///     if (t1 < 1.0)
///       split_inplace_left(wptr, t1, dim, d, p_order);
///
///     // Left endpoint relative to the new subinterval.
///     if (t1 > 0.0) t0 /= t1;
///
///     // Split right, using left endpoint.
///     if (t0 > 0.0)
///       split_inplace_right(wptr, t0, dim, d, p_order);
///   }
/// }





}//namespace dray

#endif//DRAY_ELEM_OPS_HPP
