// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ISO_OPS_HPP
#define DRAY_ISO_OPS_HPP

#include <dray/types.hpp>
#include <dray/math.hpp>
#include <dray/data_model/element.hpp>
#include <dray/data_model/dof_access.hpp>
#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/elem_ops.hpp>
#include <dray/data_model/subref.hpp>

#include <bitset>

namespace dray
{


// ----------------------------------------------------
// Isosurfacing approach based on
//   https://dx.doi.org/10.1016/j.cma.2016.10.019
//
// @article{FRIES2017759,
//   title = "Higher-order meshing of implicit geometries—Part I: Integration and interpolation in cut elements",
//   journal = "Computer Methods in Applied Mechanics and Engineering",
//   volume = "313",
//   pages = "759 - 784",
//   year = "2017",
//   issn = "0045-7825",
//   doi = "https://doi.org/10.1016/j.cma.2016.10.019",
//   url = "http://www.sciencedirect.com/science/article/pii/S0045782516308696",
//   author = "T.P. Fries and S. Omerović and D. Schöllhammer and J. Steidl",
// }
// ----------------------------------------------------

  // internal, will be undef'd at end of file.
#ifdef DRAY_CUDA_ENABLED
#define THROW_LOGIC_ERROR(msg) assert(!(msg) && false);
#elif defined(DRAY_HIP_ENABLED)
// TODO -- is this correct for HIP?
#define THROW_LOGIC_ERROR(msg) assert(!(msg) && false);
#else
#define THROW_LOGIC_ERROR(msg) throw std::logic_error(msg);
#endif


  namespace eops
  {

  //
  // Section 3.1 Valid level-set data and recursion.
  //   - Criteria defining whether further splits are needed
  //     before the isocut is considered 'simple'.
  //
  //   - The authors use a sampling approach to determine whether
  //     an element/face/edge is cut.
  //
  //   - We can use conservative Bernstein estimates instead.
  //
  //   2D:
  //     (i) Each element edge is only cut once;  |  Bounded by #(+/-) changes
  //                                              |  ('variation diminishing').
  //
  //     (ii) The overall number of cut           |  Number of edges whose bounds
  //          edges must be two;                  |  contain $\iota$ must be two.
  //
  //     (iii) If no edge is cut then             |  Edge bounds contain $\iota$
  //           the element is completely uncut.   |  or element bounds don't either.
  //
  //   3D:
  //     (i)--(iii) must hold on each face;
  //
  //     (iv) If no face is cut then              |  Face bounds contain $\iota$
  //          the element is completely uncut.    |  or element bounds don't either.
  //
  //   Based on these criteria, we should define a method to check
  //   if they are all satisfied for a given element. If one or
  //   more criteria are violated, suggest an effective split
  //   to resolve the violation.


  using ScalarDP = ReadDofPtr<Vec<Float, 1>>;

  // NOTE: When I tested this filter against the low-order conduit
  //       braid example (isoval=1), the epsilon was too large to
  //       compute a valid isosurface. This caused a crash in
  //       reconstruct_isopatch(ShapeHex,...) because it would try
  //       to write 5 edges to a stack array of size 4.
  template <typename T> DRAY_EXEC T iso_epsilon ()
  {
    return epsilon<T>();
  }

  template <> DRAY_EXEC float32 iso_epsilon<float32> ()
  {
    return 1e-8f;
  }

  template <> DRAY_EXEC float64 iso_epsilon<float64> ()
  {
    return 1e-16f;
  }

  DRAY_EXEC int8 isosign(Float value, Float isovalue)
  {
    return (value < isovalue - iso_epsilon<Float>() ? -1
            : value > isovalue + iso_epsilon<Float>() ? +1
            : +1);  // Symbolic perturbation
  }

  // TODO use sampling because the control points will be too generous.
  template <class RotatedIndexT>
  DRAY_EXEC int32 edge_var(const RotatedIndexT &wheel, const ScalarDP &dofs, Float iota, int32 p)
  {
    int32 count = 0;
    // TODO review watertight isosurfaces, what to do when equal.
    int8 prev_s = isosign(dofs[wheel.linearize(0)][0], iota);
    for (int32 i = 1; i <= p; ++i)
    {
      int8 new_s = isosign(dofs[wheel.linearize(i)][0], iota);
      if (prev_s && new_s && (new_s != prev_s))
      {
        prev_s = new_s;
        count++;
      }
    }
    return count;
  }

  template <class RotatedIndexT>
  DRAY_EXEC bool face_cut_hex(const RotatedIndexT &wheel, const ScalarDP &dofs, Float iota, int32 p)
  {
    Range dof_range;
    for (int j = 0; j <=p; ++j)
      for (int i = 0; i <=p; ++i)
        dof_range.include(dofs[wheel.linearize(i,j)][0]);
    return dof_range.contains(iota);
  }

  DRAY_EXEC bool int_cut_hex(const ScalarDP &dofs, Float iota, int32 p)
  {
    Range dof_range;
    const int32 ndofs = (p+1)*(p+1)*(p+1);
    for (int i = 0; i < ndofs; ++i)
      dof_range.include(dofs[i][0]);
    return dof_range.contains(iota);
  }


  struct IsocutInfo
  {
    enum CutOptions { Cut = 1u,         CutSimpleTri = 2u, CutSimpleQuad = 4u,
                      IntNoFace = 8u,   IntManyFace = 16u,
                      FaceNoEdge = 32u, FaceManyEdge = 64u,
                      EdgeManyPoint = 128u };
    uint8 m_cut_type_flag;
    uint8 m_bad_faces_flag;
    uint32 m_bad_edges_flag;

    DRAY_EXEC void clear() { m_cut_type_flag = 0;  m_bad_faces_flag = 0;  m_bad_edges_flag = 0; }
  };
  std::ostream & operator<<(std::ostream &out, const IsocutInfo &ici);

  struct CutEdges
  {
    uint32 cut_edges;
    uint32 complex_edges;
  };

  DRAY_EXEC CutEdges get_cut_edges(ShapeHex, const ScalarDP & dofs, Float iota, int32 p)
  {
    const HexFlat hlin{p};

    using namespace hex_flags;

    // All cut edges and bad edges (bad = cut more than once).
    uint32 ce = 0u;
    uint32 be = 0u;
    int32 ev;

    // X aligned edges
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,0,0, hlin), dofs, iota, p);
      ce |= e00 * (ev > 0);
      be |= e00 * (ev > 1);
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,p,0, hlin), dofs, iota, p);
      ce |= e01 * (ev > 0);
      be |= e01 * (ev > 1);
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,0,p, hlin), dofs, iota, p);
      ce |= e02 * (ev > 0);
      be |= e02 * (ev > 1);
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,p,p, hlin), dofs, iota, p);
      ce |= e03 * (ev > 0);
      be |= e03 * (ev > 1);

    // Y aligned edges
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(0,0,0, hlin), dofs, iota, p);
      ce |= e04 * (ev > 0);
      be |= e04 * (ev > 1);
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(p,0,0, hlin), dofs, iota, p);
      ce |= e05 * (ev > 0);
      be |= e05 * (ev > 1);
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(0,0,p, hlin), dofs, iota, p);
      ce |= e06 * (ev > 0);
      be |= e06 * (ev > 1);
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(p,0,p, hlin), dofs, iota, p);
      ce |= e07 * (ev > 0);
      be |= e07 * (ev > 1);

    // Z aligned edges
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(0,0,0, hlin), dofs, iota, p);
      ce |= e08 * (ev > 0);
      be |= e08 * (ev > 1);
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(p,0,0, hlin), dofs, iota, p);
      ce |= e09 * (ev > 0);
      be |= e09 * (ev > 1);
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(0,p,0, hlin), dofs, iota, p);
      ce |= e10 * (ev > 0);
      be |= e10 * (ev > 1);
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(p,p,0, hlin), dofs, iota, p);
      ce |= e11 * (ev > 0);
      be |= e11 * (ev > 1);

    CutEdges edge_flags;
    edge_flags.cut_edges = ce;
    edge_flags.complex_edges = be;
    return edge_flags;
  }


  DRAY_EXEC uint8 get_cut_faces(ShapeHex, const ScalarDP & dofs, Float iota, int32 p)
  {
    const HexFlat hlin{p};
    using namespace hex_flags;

    uint8 cf = 0;
    cf |= f04 * face_cut_hex(RotatedIdx3<0,1,2, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f05 * face_cut_hex(RotatedIdx3<0,1,2, HexFlat>(0,0,p, hlin), dofs, iota, p);
    cf |= f00 * face_cut_hex(RotatedIdx3<1,2,0, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f01 * face_cut_hex(RotatedIdx3<1,2,0, HexFlat>(p,0,0, hlin), dofs, iota, p);
    cf |= f02 * face_cut_hex(RotatedIdx3<2,0,1, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f03 * face_cut_hex(RotatedIdx3<2,0,1, HexFlat>(0,p,0, hlin), dofs, iota, p);

    return cf;
  }


  DRAY_EXEC IsocutInfo measure_isocut(ShapeHex, const ScalarDP & dofs, Float iota, int32 p)
  {
    IsocutInfo info;
    info.clear();

    using namespace hex_flags;

    // All cut edges and "bad" edges (bad = cut more than once).
    CutEdges edge_flags = get_cut_edges(ShapeHex(), dofs, iota, p);
    const uint32 &ce = edge_flags.cut_edges;
    const uint32 &be = edge_flags.complex_edges;

    // Update info with edges.
    info.m_bad_edges_flag = be;
    info.m_cut_type_flag |= IsocutInfo::EdgeManyPoint * bool(be);

    // All cut faces.
    const uint8 cf = get_cut_faces(ShapeHex(), dofs, iota, p);

    // FaceNoEdge (A face that is cut without any of its edges being cut).
    uint8 fne = 0;
    fne |= f04 * ((cf & f04) && !(ce & (e00 | e01 | e04 | e05)));
    fne |= f05 * ((cf & f05) && !(ce & (e02 | e03 | e06 | e07)));
    fne |= f00 * ((cf & f00) && !(ce & (e04 | e06 | e08 | e10)));
    fne |= f01 * ((cf & f01) && !(ce & (e05 | e07 | e09 | e11)));
    fne |= f02 * ((cf & f02) && !(ce & (e00 | e02 | e08 | e09)));
    fne |= f03 * ((cf & f03) && !(ce & (e01 | e03 | e10 | e11)));

    // FaceManyEdge (A face for which more than two incident edges are cut).
    uint8 fme = 0;
    fme |= f04 * (bool(ce & e00) + bool(ce & e01) + bool(ce & e04) + bool(ce & e05) > 2);
    fme |= f05 * (bool(ce & e02) + bool(ce & e03) + bool(ce & e06) + bool(ce & e07) > 2);
    fme |= f00 * (bool(ce & e04) + bool(ce & e06) + bool(ce & e08) + bool(ce & e10) > 2);
    fme |= f01 * (bool(ce & e05) + bool(ce & e07) + bool(ce & e09) + bool(ce & e11) > 2);
    fme |= f02 * (bool(ce & e00) + bool(ce & e02) + bool(ce & e08) + bool(ce & e09) > 2);
    fme |= f03 * (bool(ce & e01) + bool(ce & e03) + bool(ce & e10) + bool(ce & e11) > 2);

    // Update info with faces.
    info.m_bad_faces_flag |= fne | fme;
    info.m_cut_type_flag |= IsocutInfo::FaceNoEdge * bool(fne);
    info.m_cut_type_flag |= IsocutInfo::FaceManyEdge * bool(fme);

    const int8 num_cut_faces
      = (uint8(0) + bool(cf & f04) + bool(cf & f05) + bool(cf & f00)
                  + bool(cf & f01) + bool(cf & f02) + bool(cf & f03));

    const bool ci = int_cut_hex(dofs, iota, p);

    // Update info with interior.
    info.m_cut_type_flag |= IsocutInfo::IntNoFace * (ci && !cf);
    info.m_cut_type_flag |= IsocutInfo::IntManyFace * (num_cut_faces > 4);

    // Cut or not.
    info.m_cut_type_flag |= IsocutInfo::Cut * (ci || cf || ce);

    // Combine all info to describe whether the cut is simple.
    if (info.m_cut_type_flag < 8)
    {
      info.m_cut_type_flag |= IsocutInfo::CutSimpleTri *  (num_cut_faces == 3);
      info.m_cut_type_flag |= IsocutInfo::CutSimpleQuad * (num_cut_faces == 4);
    }

    return info;
  }


  DRAY_EXEC Split<Tensor> pick_iso_simple_split(ShapeHex, const IsocutInfo &info)
  {
    using namespace hex_flags;

    const uint8 &bf = info.m_bad_faces_flag;
    const uint32 &be = info.m_bad_edges_flag;

    Float score_x = 0, score_y = 0, score_z = 0;

    // Problematic edges on an axis increase likelihood that axis is split.
    score_x += 0.25f * (bool(be & e00) + bool(be & e01) + bool(be & e02) + bool(be & e03));
    score_y += 0.25f * (bool(be & e04) + bool(be & e05) + bool(be & e06) + bool(be & e07));
    score_z += 0.25f * (bool(be & e08) + bool(be & e09) + bool(be & e10) + bool(be & e11));

    // Problematic faces normal to an axis decrease likelihood that axis is split.
    score_x -= 0.5f * (bool(bf & f00) + bool(bf & f01));
    score_y -= 0.5f * (bool(bf & f02) + bool(bf & f03));
    score_z -= 0.5f * (bool(bf & f04) + bool(bf & f05));

    const int32 split_axis = (score_x > score_y ?
                               (score_x > score_z ? 0 : 2) :
                               (score_y > score_z ? 1 : 2));

    return Split<Tensor>::half(split_axis);
  }




  DRAY_EXEC IsocutInfo measure_isocut(ShapeTet, const ScalarDP & dofs, Float iota, int32 p)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " measure_isocut(ShapeTet)")
    return *(IsocutInfo*)nullptr;
  }


  DRAY_EXEC Split<Simplex> pick_iso_simple_split(ShapeTet, const IsocutInfo &info)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " pick_iso_simple_split(ShapeTet)")
    return *(Split<Simplex>*)nullptr;
  }



  // TODO These routines might not handle degeneracies gracefully. Need symbolic perturbations.

  DRAY_EXEC Float isointercept_linear(const Vec<Float, 1> &v0,
                                      const Vec<Float, 1> &v1,
                                      Float iota)
  {
    // Assume there exists t such that:
    //     v0 * (1-t) + v1 * (t) == iota
    //     <--> t * (v1-v0) == (iota-v0)
    //     <--> t = (iota-v0)/(v1-v0)  or  v0==v1
    const Float delta = v1[0] - v0[0];
    iota -= v0[0];
    return iota / delta;
  }

  DRAY_EXEC Float isointercept_quadratic(const Vec<Float, 1> &v0,
                                         const Vec<Float, 1> &v1,
                                         const Vec<Float, 1> &v2,
                                         Float iota)
  {
    // Assume there exists t such that:
    //     v0 * (1-t)^2  +  v1 * 2*(1-t)*t  +  v2 * (t)^2  == iota
    //     <--> t^2 * (v2 - 2*v1 + v0) + t * 2*(v1 - v0) == (iota-v0)
    //              dd20:=(v2 - 2*v1 + v0)      d10:=(v1-v0)
    //
    //      --> t = -(d10/dd20) +/- sqrt[(iota-v0)/dd20 + (d10/dd20)^2]
    //
    const Float d10 = v1[0] - v0[0];
    const Float dd20 = v2[0] - 2*v1[0] + v0[0];
    iota -= v0[0];
    const Float x = -d10/dd20;
    const Float w = sqrt(iota/dd20 + (x*x));
    const Float tA = x+w;
    const Float tB = x-w;
    // If one root is in the unit interval, pick it.
    return (fabs(tA-0.5) <= fabs(tB-0.5) ? tA : tB);
  }


  DRAY_EXEC Float cut_edge_hex(const uint8 eid, const ScalarDP &C, Float iota, const OrderPolicy<1> order_p)
  {
    constexpr uint8 p = eattr::get_order(order_p.as_cxp());
    const int32 off0 = hex_props::hex_eoffset0(eid);
    const int32 off1 = hex_props::hex_eoffset1(eid);
    const int32 off2 = hex_props::hex_eoffset2(eid);
    const int32 offset = p*(p+1)*(p+1)*off2 + p*(p+1)*off1 + p*off0;
    const int32 stride = hex_props::hex_estride(eid, p+1);

    return isointercept_linear(C[offset + 0*stride], C[offset + 1*stride], iota);
  }

  DRAY_EXEC Float cut_edge_hex(const uint8 eid, const ScalarDP &C, Float iota, const OrderPolicy<2> order_p)
  {
    constexpr uint8 p = eattr::get_order(order_p.as_cxp());
    const int32 off0 = hex_props::hex_eoffset0(eid);
    const int32 off1 = hex_props::hex_eoffset1(eid);
    const int32 off2 = hex_props::hex_eoffset2(eid);
    const int32 offset = p*(p+1)*(p+1)*off2 + p*(p+1)*off1 + p*off0;
    const int32 stride = hex_props::hex_estride(eid, p+1);

    return isointercept_quadratic(C[offset + 0*stride],
                                  C[offset + 1*stride],
                                  C[offset + 2*stride], iota);
  }

  DRAY_EXEC Float cut_edge_hex(const uint8 eid, const ScalarDP &C, Float iota, const OrderPolicy<General> order_p)
  {
    const int32 p = eattr::get_order(order_p);
    const HexEdgeWalker<General> hew(order_p, eid);

    // Initial guess should be near the crossing.
    // If this function is called on a 'simple isocut,' then we know
    // there is exactly one crossing and exactly one sign change.
    int8 v_lo = 0, v_hi = p;
    const bool sign_lo = C[hew.edge2hex(v_lo)][0] >= iota;
    const bool sign_hi = C[hew.edge2hex(v_hi)][0] >= iota;
    while (v_lo < p && (C[hew.edge2hex(v_lo+1)][0] >= iota) == sign_lo)
      v_lo++;
    while (v_hi > 0 && (C[hew.edge2hex(v_hi-1)][0] >= iota) == sign_hi)
      v_hi--;
    Vec<Float, 1> r = {{0.5f * (v_lo + v_hi) / p}};

    Float scalar;
    Vec<Vec<Float, 1>, 1> deriv;

    // Do num_iter Newton--Raphson steps.
    const int8 num_iter = 8;
    for (int8 step = 0; step < num_iter; step++)
    {
      scalar = eval_d_edge(ShapeHex(), order_p, eid, C, r, deriv)[0];
      r[0] += (iota-scalar)/deriv[0][0];
    }

    return r[0];
  }


  /**
   * @brief Solve for the reference coordinates of a triangular isopatch inside a hex.
   */
  template <int32 IP, int32 OP>
  DRAY_EXEC void reconstruct_isopatch(ShapeHex, ShapeTri,
      const ScalarDP & in,
      WriteDofPtr<Vec<Float, 3>> & out,
      Float iota,
      OrderPolicy<IP> in_order_p,
      OrderPolicy<OP> out_order_p)
  {
    // Since the isocut is 'simple,' there is a very restricted set of cases.
    // Each cut face has exactly two cut edges: Cell faces -> patch edges.
    // For tri patch, there are 3 cut faces and 3 cut edges.
    // Among the cut faces, each pair must share an edge. Opposing faces eliminated.
    // Thus from (6 choose 3)==20 face combos, 12 are eliminated, leaving 8.
    // These 8 correspond to 3sets joined by a common vertex.
    //
    // 000: X0.Y0.Z0     (f2.f4.f0)
    // 001: X1.Y0.Z0     (f3.f4.f0)
    // ...
    // 111: X1.Y1.Z1     (f3.f5.f1)

    using namespace hex_flags;

    /// constexpr uint8 caseF000 = f00|f02|f04;  constexpr uint32 caseE000 = e00|e04|e08;
    /// constexpr uint8 caseF001 = f01|f02|f04;  constexpr uint32 caseE001 = e00|e05|e09;
    /// constexpr uint8 caseF010 = f00|f03|f04;  constexpr uint32 caseE010 = e01|e04|e10;
    /// constexpr uint8 caseF011 = f01|f03|f04;  constexpr uint32 caseE011 = e01|e05|e11;
    /// constexpr uint8 caseF100 = f00|f02|f05;  constexpr uint32 caseE100 = e02|e06|e08;
    /// constexpr uint8 caseF101 = f01|f02|f05;  constexpr uint32 caseE101 = e02|e07|e09;
    /// constexpr uint8 caseF110 = f00|f03|f05;  constexpr uint32 caseE110 = e03|e06|e10;
    /// constexpr uint8 caseF111 = f01|f03|f05;  constexpr uint32 caseE111 = e03|e07|e11;

    const int32 iP = eattr::get_order(in_order_p);
    const int32 oP = eattr::get_order(out_order_p);

    const uint32 cut_edges = get_cut_edges(ShapeHex(), in, iota, iP).cut_edges;
    const uint8 cut_faces = get_cut_faces(ShapeHex(), in, iota, iP);

    using ::dray::detail::cartesian_to_tri_idx;

    // For each cell edge, solve for isovalue intercept along the edge.
    // This is univariate root finding for an isolated single root.
    // --> Vertices of the isopatch.
    uint8 edge_ids[3];
    Float edge_split[3];

    if (!(cut_edges & (e00 | e01 | e02 | e03)))
      THROW_LOGIC_ERROR("Hex->Tri: No X edges (" __FILE__ ")")
    if (!(cut_edges & (e04 | e05 | e06 | e07)))
      THROW_LOGIC_ERROR("Hex->Tri: No Y edges (" __FILE__ ")")
    if (!(cut_edges & (e08 | e09 | e10 | e11)))
      THROW_LOGIC_ERROR("Hex->Tri: No Z edges (" __FILE__ ")")

    edge_ids[0] = (cut_edges & e00) ? 0 : (cut_edges & e01) ? 1 : (cut_edges & e02) ? 2 : 3;
    edge_ids[1] = (cut_edges & e04) ? 4 : (cut_edges & e05) ? 5 : (cut_edges & e06) ? 6 : 7;
    edge_ids[2] = (cut_edges & e08) ? 8 : (cut_edges & e09) ? 9 : (cut_edges & e10) ? 10 : 11;

    edge_split[0] = cut_edge_hex(edge_ids[0], in, iota, in_order_p);
    edge_split[1] = cut_edge_hex(edge_ids[1], in, iota, in_order_p);
    edge_split[2] = cut_edge_hex(edge_ids[2], in, iota, in_order_p);

    // triW:edge_ids[0]  triX:edge_ids[1]  triY:edge_ids[2]
    const Vec<Float, 3> vW = {{edge_split[0], 1.0f*bool(cut_edges & (e01 | e03)), 1.0f*bool(cut_edges & (e02 | e03))}};
    const Vec<Float, 3> vX = {{1.0f*bool(cut_edges & (e05 | e07)), edge_split[1], 1.0f*bool(cut_edges & (e06 | e07))}};
    const Vec<Float, 3> vY = {{1.0f*bool(cut_edges & (e09 | e11)), 1.0f*bool(cut_edges & (e10 | e11)), edge_split[2]}};

    out[cartesian_to_tri_idx(0,0,oP+1)] = vW;
    out[cartesian_to_tri_idx(oP,0,oP+1)] = vX;
    out[cartesian_to_tri_idx(0,oP,oP+1)] = vY;


    // For each cell face, solve for points in middle of isocontour within the face.
    // --> Boundary edges the isopatch.

    // Set initial guesses for patch edges (linear).
    for (uint8 i = 1; i < oP; ++i)
    {
      out[cartesian_to_tri_idx(i, 0, oP+1)]    = (vW*(oP-i) + vX*i)/oP;   // Tri edge W-->0
    }
    for (uint8 i = 1; i < oP; ++i)
    {
      out[cartesian_to_tri_idx(0, i, oP+1)]    = (vW*(oP-i) + vY*i)/oP;   // Tri edge W-->1
      out[cartesian_to_tri_idx(oP-i, i, oP+1)] = (vX*(oP-i) + vY*i)/oP;   // Tri edge 0-->1
    }


    // Solve for edge interiors.
    for (uint8 patch_edge_idx = 0; patch_edge_idx < 3; ++patch_edge_idx)
    {
      // Need patch edge id and cell face id to convert coordinates.
      constexpr tri_props::EdgeIds edge_list[3] = { tri_props::edgeW0,
                                                    tri_props::edgeW1,
                                                    tri_props::edge01 };
      const uint8 patch_edge = edge_list[patch_edge_idx];

      constexpr uint8 te_end0[3] = {0, 0, 1};
      constexpr uint8 te_end1[3] = {1, 2, 2};

      const uint8 fid = hex_props::hex_common_face(
          edge_ids[te_end0[patch_edge_idx]], edge_ids[te_end1[patch_edge_idx]] );
      const HexFaceWalker<IP> cell_fw(in_order_p, fid);
      const TriEdgeWalker<OP> patch_ew(out_order_p, patch_edge);

      // Solve for each dof in the edge.
      for (int32 i = 1; i < oP; ++i)
      {
        // Get initial guess.
        Vec<Float, 3> pt3 = out[patch_ew.edge2tri(i)];
        Vec<Float, 2> pt2 = {{pt3[hex_props::hex_faxisU(fid)],
                              pt3[hex_props::hex_faxisV(fid)]}};

        // Option 14: Use (initial) gradient direction as search.
        Vec<Float, 2> search_dir;
        {
          // For the search direction, use initial gradient direction (variant 14).
          Vec<Vec<Float, 1>, 2> deriv;
          Vec<Float, 1> scalar = eval_d_face(ShapeHex(), in_order_p, fid, in, pt2, deriv);

          search_dir = (Vec<Float, 2>{{ deriv[0][0], deriv[1][0]}}).normalized();
        }

        // Do a few iterations.
        constexpr int32 num_iter = 5;
        for (int32 t = 0; t < num_iter; ++t)
        {
          Vec<Vec<Float, 1>, 2> deriv;
          Vec<Float, 1> scalar = eval_d_face(ShapeHex(), in_order_p, fid, in, pt2, deriv);
          pt2 += search_dir * ((iota-scalar[0]) * rcp_safe(dot(deriv, search_dir)[0]));
        }

        // Store the updated point.
        pt3[hex_props::hex_faxisU(fid)] = pt2[0];
        pt3[hex_props::hex_faxisV(fid)] = pt2[1];
        out[patch_ew.edge2tri(i)] = pt3;
      }
    }


    // For the cell volume, solve for points in middle of isopatch.

    // Initial guess for patch interior.
    // Based on paper appendix formula for triangles, with modifications.
    //
    // Same basic idea:
    // - Subtract linear part of each edge function;
    // - Use rational ramp functions to interpolate nonlinear parts to patch interior;
    //   - One ramp function per edge, which takes value 1 on that edge, 0 on other edges.
    //   - Contains poles at triangle vertices, but the nonlinear part is zero there,
    //       and we don't need to interpolate corners anyway.
    // - Add back the linear part over the interior.
    //
    // I have two small modifications:
    //   (1.) In order to evaluate the ramp function in the triangle interior,
    //        the 2D barycentric coordinates must be related to the 1D edge coordinate
    //        along each edge.  This relation can be accomplished through a projection.
    //        The original method uses an orthogonal projection onto edges,
    //        assuming the geometry of a right triangle, which is not symmetric.
    //        I will also project orthogonally onto edges, but I will assume
    //        the geometry of an equilateral triangle, which is symmetric.
    //
    //   (2.) The original method evaluates the nonlinear part of each edge function,
    //        using polynomial interpolation over all known edge points.
    //        I will approximate this by, after projecting to the edge, finding the
    //        one or two nearest known points, and interpolating linearly between them.
    //
    // TODO Modification (2.) should be removed later in favor of full polynomial interpolation,
    // but Lagrange/Newton evaluation is not supported yet.

    {
      const Vec<Float, 3> vp00 = out[cartesian_to_tri_idx(oP, 0, oP+1)];
      const Vec<Float, 3> v0p0 = out[cartesian_to_tri_idx(0, oP, oP+1)];
      const Vec<Float, 3> v00p = out[cartesian_to_tri_idx(0, 0, oP+1)];

      // Orthogonal projections: Follow parallelograms
      //      [  i  j mu ] (indices)
      //      [  u  v  t ] (coords)
      //   --> (-2,+1,+1) till meet (i=0) edge;
      //   --> (+1,-2,+1) till meet (j=0) edge;
      //   --> (+1,+1,-2) till meet (mu=0) edge.
      const Vec<int32, 3> toi0  = {{-2,  1,  1}};
      const Vec<int32, 3> toj0  = {{ 1, -2,  1}};
      const Vec<int32, 3> tomu0 = {{ 1,  1, -2}};

      for (int32 j = 1; j < oP; ++j)
        for (int32 i = 1; i < oP-j; ++i)
        {
          const int32 mu = oP-j-i;
          const Float u = (Float(i)/Float(oP));
          const Float v = (Float(j)/Float(oP));
          const Float t = (Float(mu)/Float(oP));

          // Initially, linear part.
          Vec<Float, 3> outval = vp00 * u + v0p0 * v + v00p * t;

          Vec<int32, 3> bc = {{i, j, mu}};    // Store barycentric coords near edges.
          Vec<Float, 3> nl;                   // Calculate nonlinear part on edge.
          Float e;                            // Edge coordinate.
          int32 count_prllgm;
          Float ramp;

          // Nonlinear contribution from (i=0) edge [(u=0) edge].
          count_prllgm = i / 2;
          bc += toi0 * count_prllgm;
          e = (bc[0] == 0 ? Float(bc[1])/Float(oP) : Float(bc[1]+0.5f)/Float(oP)); //+:t->v
          nl = out[cartesian_to_tri_idx(0, bc[1], oP+1)];
          if (bc[0] > 0)
            nl = nl * 0.5 + out[cartesian_to_tri_idx(0, bc[1]+1, oP+1)] * 0.5; //TODO
          nl -= v0p0 * e + v00p * (1.0f-e);       // Subtract edge linear part.
          ramp = ((v*t) / (e*(1.0f-e)));
          outval += nl * ramp;
          bc -= toi0 * count_prllgm;

          // Nonlinear contribution from (j=0) edge [(v=0) edge].
          count_prllgm = j / 2;
          bc += toj0 * count_prllgm;
          e = (bc[1] == 0 ? Float(bc[0])/Float(oP) : Float(bc[0]+0.5f)/Float(oP)); //+:t->u
          nl = out[cartesian_to_tri_idx(bc[0], 0, oP+1)];
          if (bc[1] > 0)
            nl = nl * 0.5 + out[cartesian_to_tri_idx(bc[0]+1, 0, oP+1)] * 0.5; //TODO
          nl -= vp00 * e + v00p * (1.0f-e);       // Subtract edge linear part.
          ramp = ((u*t) / (e*(1.0f-e)));
          outval += nl * ramp;
          bc -= toj0 * count_prllgm;

          // Nonlinear contribution from (mu=0) edge [(t=0) edge].
          count_prllgm = mu / 2;
          bc += tomu0 * count_prllgm;
          e = (bc[2] == 0 ? Float(bc[1])/Float(oP) : Float(bc[1]+0.5f)/Float(oP)); //+:u->v
          nl = out[cartesian_to_tri_idx(oP-bc[1], bc[1], oP+1)];
          if (bc[2] > 0)
            nl = nl * 0.5 + out[cartesian_to_tri_idx(oP-(bc[1]+1), bc[1]+1, oP+1)] * 0.5; //TODO
          nl -= v0p0 * e + vp00 * (1.0f-e);       // Subtract edge linear part.
          ramp = ((u*v) / (e*(1.0f-e)));
          outval += nl * ramp;
          bc -= tomu0 * count_prllgm;

          out[cartesian_to_tri_idx(i, j, oP+1)] = outval;
        }
    }

    // Solve for patch interior.
    // TODO coordination for optimal spacing.
    // For now, move each point individually.
    for (int32 j = 1; j < oP; ++j)
      for (int32 i = 1; i < oP-j; ++i)
      {
        // Get initial guess.
        Vec<Float, 3> pt3 = out[cartesian_to_tri_idx(i, j, (oP+1))];

        Vec<Vec<Float, 1>, 3> deriv;
        Vec<Float, 1> scalar = eval_d(ShapeHex(), in_order_p, in, pt3, deriv);

        // For the search direction, use initial gradient direction.
        const Vec<Float, 3> search_dir =
            (Vec<Float, 3>{{deriv[0][0], deriv[1][0], deriv[2][0]}}).normalized();

        // Do a few iterations.
        constexpr int32 num_iter = 5;
        for (int32 t = 0; t < num_iter; ++t)
        {
          scalar = eval_d(ShapeHex(), in_order_p, in, pt3, deriv);
          pt3 += search_dir * ((iota-scalar[0]) * rcp_safe(dot(deriv, search_dir)[0]));
        }

        // Store the updated point.
        pt3 = out[cartesian_to_tri_idx(i, j, (oP+1))] = pt3;
      }

    /// switch (cut_faces)
    /// {
    ///   case caseF000: assert(cut_edges == caseE000);
    ///     break;
    ///   case caseF001: assert(cut_edges == caseE001);
    ///     break;
    ///   case caseF010: assert(cut_edges == caseE010);
    ///     break;
    ///   case caseF011: assert(cut_edges == caseE011);
    ///     break;
    ///   case caseF100: assert(cut_edges == caseE100);
    ///     break;
    ///   case caseF101: assert(cut_edges == caseE101);
    ///     break;
    ///   case caseF110: assert(cut_edges == caseE110);
    ///     break;
    ///   case caseF111: assert(cut_edges == caseE111);
    ///     break;
    ///   default:
    ///     THROW_LOGIC_ERROR("Unexpected tri isopatch case (" __FILE__ ")")
    /// }

    // TODO consider breaking out the solves for each point for higher parallelism.

  }


  /**
   * @brief Solve for the reference coordinates of a quad isopatch inside a hex.
   */
  template <int32 IP, int32 OP>
  DRAY_EXEC void reconstruct_isopatch(ShapeHex, ShapeQuad,
      const ScalarDP & in,
      WriteDofPtr<Vec<Float, 3>> & out,
      Float iota,
      OrderPolicy<IP> in_order_p,
      OrderPolicy<OP> out_order_p)
  {
    // Since the isocut is 'simple,' there is a very restricted set of cases.
    // Each cut face has exactly two cut edges: Cell faces -> patch edges.
    // For quad patch, there are 4 cut faces and 4 cut edges.
    // All (6 choose 4)==15 combos are valid cuts.
    // There are two types: Axis-aligned into 2 cubes --> x3 axes
    //                      Corner-cutting into prism --> x4 corners x3 axes.

    const int32 iP = eattr::get_order(in_order_p);
    const int32 oP = eattr::get_order(out_order_p);

    const uint32 cut_edges = get_cut_edges(ShapeHex(), in, iota, iP).cut_edges;
    const uint8 cut_faces = get_cut_faces(ShapeHex(), in, iota, iP);

    using namespace hex_flags;

    uint8 edge_ids[4];
    uint8 split_counter = 0;
    for (uint8 e = 0; e < 12; ++e)
    {
      if ((cut_edges & (1u<<e)))
      {
#ifdef DEBUG_ISOSURFACE_FILTER
        if(split_counter >= 4)
        {
          split_counter++;
          continue;
        }
#endif
        edge_ids[split_counter++] = e;
      }
    }
#ifdef DEBUG_ISOSURFACE_FILTER
    if(split_counter > 4)
    {
      std::cout << "ERROR: split_counter=" << split_counter;
                << "\nEdge cases: \n  ";
      for(uint8 e = 0; e < 12; ++e)
      {
        std::cout << int(cut_edges & (1u<<e)) << " ";
      }
      std::cout << "\nPoint values: \n  ";
      for(uint8 dof = 0; dof < 8; ++dof)
      {
        std::cout << in[dof][0] << " ";
      }
      std::cout << std::endl;
    }
#endif

    // Corners of the patch live on cell edges.
    Vec<Float, 3> corners[4];
    for (uint8 s = 0; s < 4; ++s)
    {
      const uint8 e = edge_ids[s];

      // Get base coordinate of cell edge.
      corners[s][0] = hex_props::hex_eoffset0(e);
      corners[s][1] = hex_props::hex_eoffset1(e);
      corners[s][2] = hex_props::hex_eoffset2(e);

      // Overwrite coordinate along the cell edge based on iso cut.
      corners[s][hex_props::hex_eaxis(e)] = cut_edge_hex(e, in, iota, in_order_p);
    }

    out[(oP+1)*0 + 0]   = corners[0];
    out[(oP+1)*0 + oP]  = corners[1];
    out[(oP+1)*oP + 0]  = corners[2];
    out[(oP+1)*oP + oP] = corners[3];

    // Set initial guesses for patch edges (linear).
    for (uint8 i = 1; i < oP; ++i)
    {
      out[(oP+1)*0 + i] = (corners[0]*(oP-i) + corners[1]*i)/oP;  // Quad edge 0
    }
    for (uint8 i = 1; i < oP; ++i)
    {
      out[(oP+1)*i + 0]  = (corners[0]*(oP-i) + corners[2]*i)/oP;  // Quad edge 2
      out[(oP+1)*i + oP] = (corners[1]*(oP-i) + corners[3]*i)/oP;  // Quad edge 3
    }
    for (uint8 i = 1; i < oP; ++i)
    {
      out[(oP+1)*oP + i] = (corners[2]*(oP-i) + corners[3]*i)/oP;  // Quad edge 1
    }


    // Solve for edge interiors.
    for (uint8 patch_edge = 0; patch_edge < 4; ++patch_edge)
    {
      constexpr uint8 qe_end0[4] = {0, 2, 0, 1};
      constexpr uint8 qe_end1[4] = {1, 3, 2, 3};

      const uint8 fid = hex_props::hex_common_face(
          edge_ids[qe_end0[patch_edge]], edge_ids[qe_end1[patch_edge]] );
      const HexFaceWalker<IP> cell_fw(in_order_p, fid);
      const QuadEdgeWalker<OP> patch_ew(out_order_p, patch_edge);

      // For now, move each point individually.
      // TODO coordination to get optimal spacing.
      for (int32 i = 1; i < oP; ++i)
      {
        // Get initial guess.
        Vec<Float, 3> pt3 = out[patch_ew.edge2quad(i)];
        Vec<Float, 2> pt2 = {{pt3[hex_props::hex_faxisU(fid)],
                              pt3[hex_props::hex_faxisV(fid)]}};

        // Option 14: Use (initial) gradient direction as search.
        Vec<Float, 2> search_dir;
        {
          Vec<Vec<Float, 1>, 2> deriv;
          Vec<Float, 1> scalar = eval_d_face(ShapeHex(), in_order_p, fid, in, pt2, deriv);

          search_dir = (Vec<Float, 2>{{deriv[0][0], deriv[1][0]}}).normalized();
        }

        // Do a few iterations.
        constexpr int32 num_iter = 5;
        for (int32 t = 0; t < num_iter; ++t)
        {
          Vec<Vec<Float, 1>, 2> deriv;
          Vec<Float, 1> scalar = eval_d_face(ShapeHex(), in_order_p, fid, in, pt2, deriv);
          pt2 += search_dir * ((iota-scalar[0]) * rcp_safe(dot(deriv, search_dir)[0]));
        }

        // Store the updated point.
        pt3[hex_props::hex_faxisU(fid)] = pt2[0];
        pt3[hex_props::hex_faxisV(fid)] = pt2[1];
        out[patch_ew.edge2quad(i)] = pt3;
      }
    }

    // Initial guess for patch interior.
    // Follows paper appendix formula for quads.

    // The paper (transformed to the unit square [0,1]^2) does this:
    //
    //  out(i,j) =  corners[0]*(1-xi)*(1-yj) + corners[1]*( xi )*(1-yj)   // Lerp corners
    //            + corners[2]*(1-xi)*( yj ) + corners[3]*( xi )*( yj )
    //
    //            + (out(i,0) - corners[0]*(1-xi) - corners[1]*( xi ))*(1-yj) // Eval deviation on x edges
    //            + (out(i,p) - corners[2]*(1-xi) - corners[3]*( xi ))*( yj ) //  and lerp the deviation
    //
    //            + (out(0,j) - corners[0]*(1-yj) - corners[2]*( yj ))*(1-xi) // Eval deviation on y edges
    //            + (out(p,j) - corners[1]*(1-yj) - corners[3]*( yj ))*( xi ) //  and lerp the deviation
    //
    // Many of the terms cancel, making this equivalent:
    //
    for (int32 j = 1; j < oP; ++j)
    {
      const Vec<Float, 3> dof_e2 = out[(oP+1)*j + 0];
      const Vec<Float, 3> dof_e3 = out[(oP+1)*j + oP];
      const Float yj = Float(j)/Float(oP);
      const Float _yj = 1.0f - yj;

      for (int32 i = 1; i < oP; ++i)
      {
        const Vec<Float, 3> dof_e0 = out[(oP+1)*0 + i];
        const Vec<Float, 3> dof_e1 = out[(oP+1)*oP + i];
        const Float xi = Float(i)/Float(oP);
        const Float _xi = 1.0f - xi;

        out[(oP+1)*j + i] =  dof_e0 * _yj + dof_e1 * yj
                          + dof_e2 * _xi + dof_e3 * xi
                          - corners[0] * (_xi * _yj)
                          - corners[1] * ( xi * _yj)
                          - corners[2] * (_xi *  yj)
                          - corners[3] * ( xi *  yj);
      }
    }

    // Solve for patch interior.
    // TODO coordination for optimal spacing.
    // For now, move each point individually.
    for (int32 j = 1; j < oP; ++j)
      for (int32 i = 1; i < oP; ++i)
      {
        // Get initial guess.
        Vec<Float, 3> pt3 = out[(oP+1)*j + i];

        Vec<Vec<Float, 1>, 3> deriv;
        Vec<Float, 1> scalar = eval_d(ShapeHex(), in_order_p, in, pt3, deriv);

        // For the search direction, use initial gradient direction.
        const Vec<Float, 3> search_dir =
            (Vec<Float, 3>{{deriv[0][0], deriv[1][0], deriv[2][0]}}).normalized();

        // Do a few iterations.
        constexpr int32 num_iter = 5;
        for (int32 t = 0; t < num_iter; ++t)
        {
          scalar = eval_d(ShapeHex(), in_order_p, in, pt3, deriv);
          pt3 += search_dir * ((iota-scalar[0]) * rcp_safe(dot(deriv, search_dir)[0]));
        }

        // Store the updated point.
        out[(oP+1)*j + i] = pt3;
      }


    // TODO consider breaking out the solves for each point for higher parallelism.

    // For each cell edge, solve for isovalue intercept along the edge.
    // This is univariate root finding for an isolated single root.
    // --> Vertices of the isopatch.

    // For each cell face, solve for points in middle of isocontour within the face.
    // --> Boundary edges the isopatch.

    // For the cell volume, solve for points in middle of isopatch.
  }


  /**
   * @brief Solve for the reference coordinates of a triangular isopatch inside a tet.
   */
  template <int32 IP, int32 OP>
  DRAY_EXEC void reconstruct_isopatch(ShapeTet, ShapeTri,
      const ScalarDP & dofs_in,
      WriteDofPtr<Vec<Float, 3>> & lagrange_pts_out,
      Float iota,
      OrderPolicy<IP> in_order_p,
      OrderPolicy<OP> out_order_p)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " reconstruct_isopatch(ShapeTet, ShapeTri)")
  }

  /**
   * @brief Solve for the reference coordinates of a quad isopatch inside a tet.
   */
  template <int32 IP, int32 OP>
  DRAY_EXEC void reconstruct_isopatch(ShapeTet, ShapeQuad,
      const ScalarDP & dofs_in,
      WriteDofPtr<Vec<Float, 3>> & lagrange_pts_out,
      Float iota,
      OrderPolicy<IP> in_order_p,
      OrderPolicy<OP> out_order_p)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " reconstruct_isopatch(ShapeTet, ShapeQuad)")
  }



  }//eops


#undef THROW_LOGIC_ERROR

}//namespace dray

#endif//DRAY_ISO_OPS_HPP
