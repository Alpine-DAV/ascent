// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/filters/to_bernstein.hpp>
#include <dray/error.hpp>
#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>

#include <dray/data_model/mesh.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/device_mesh.hpp>
#include <dray/data_model/device_field.hpp>
#include <dray/data_model/elem_attr.hpp>


/**
 * Algorithms based on (Ainsworth & Sanchez, 2016).
 *
 * @article{ainsworth2016computing,
 *   title={Computing the Bezier control points of the Lagrangian interpolant in arbitrary dimension},
 *   author={Ainsworth, Mark and S{\'a}nchez, Manuel A},
 *   journal={SIAM Journal on Scientific Computing},
 *   volume={38},
 *   number={3},
 *   pages={A1682--A1700},
 *   year={2016},
 *   publisher={SIAM}
 * }
 */

namespace dray
{

  // Dispatch topology
  // Create new dataset
  //
  // For each field,
  //   dispatch field
  //   add to dataset
  //
  //
  // Task:
  //   (Assume that no dofs are shared)
  //   Make identically shaped grid function.
  //   RAJA-for each element,
  //     Create ReadDofPtr from input gf, WriteDofPtr from output gf.
  //     1D buffers
  //     Work magic


  // ---------- Implementation ------------------- //


  /**
   * @param p Polynomial order.
   * @param scaleX Each x value will be multiplied by this scalar before use.
   * @param x Row of x-spacing of reference coordinates of interpolation points.
   * @param w Scratch buffer of size (p+1).
   * @param f Input function values at the interpolation points, will be overwritten.
   * @param c Output function control points at the Bernstein-Bezier nodes.
   */
  template <int32 ncomp>
  DRAY_EXEC void NewtonBernstein1D_scaled(const int32 p,
                                          const Float scaleX,
                                          const Float *x,
                                          Float *w,
                                          Vec<Float, ncomp> *f,
                                          Vec<Float, ncomp> *c)
  {
    for (int32 i = 0; i <= p; ++i)
      c[i] = w[i] = 0;

    c[0] = f[0];
    w[0] = 1;
    for (int32 k = 1; k <= p; ++k)
    {
      for (int32 j = p; j >= k; --j)
        f[j] = (f[j]-f[j-1]) / ((x[j] - x[j-k])*scaleX);
      for (int32 j = k; j >= 1; --j)
      {
        w[j] = w[j-1] * ((1.0f*j/k) * (1-x[k-1]*scaleX)) -  w[j] * ((1-1.0f*j/k) * x[k-1]*scaleX);
        c[j] = c[j-1] * (1.0f*j/k) + c[j] * (1-1.0f*j/k) + f[k] * w[j];
      }
      w[0] = -w[0] * x[k-1]*scaleX;
      c[0] = c[0] + f[k] * w[0];
    }
  }


  template <int32 ncomp>
  DRAY_EXEC void NewtonBernstein1D(const int32 p,
                                   const Float *x,
                                   Float *w,
                                   Vec<Float, ncomp> *f,
                                   Vec<Float, ncomp> *c)
  {
    NewtonBernstein1D_scaled(p, 1.0f, x, w, f, c);
  }



  /**
   * @param p Polynomial order.
   * @param x Row of x-spacing of reference coordinates of interpolation points.
   * @param y Column of y-spacing of reference coordinates of interpolation points.
   * @param w     Scratch buffer of size (p+1).
   * @param gprod Scratch buffer of size (p+1).
   * @param ftmp  Scratch buffer of size (p+1).
   * @param c     Scratch buffer of size (p+1).
   * @param data  [in/out] In as function vals @ interp pts, out as ctrl points @ bernstein nodes
   *              (size (n+1)*(n+2)/2).
   */
  template <int32 ncomp>
  DRAY_EXEC void NewtonBernsteinTri(const int32 p,
                                    const Float *x,
                                    const Float *y,
                                    Float *w,
                                    Float *gprod,
                                    Vec<Float, ncomp> *ftmp,
                                    Vec<Float, ncomp> *c,
                                    Vec<Float, ncomp> *data)
  {
    using detail::cartesian_to_tri_idx;

    // Initially represents the constant unit polynomial, i.e. 1.0f everywhere.
    for (int32 i = 1; i <= p; ++i)
      gprod[i] = 0;
    gprod[0] = 1.0f;

    // Solve univariate problem on each line,
    // and accumulate into triangular solution.
    // Start at bottom row (j=p, pmj=0), which contains (p+1) points.
    for (int32 j = p; j >= 1; --j)
    {
      const int32 pmj = p-j;

      // Get input row for 1D problem and replace by 0 before add output.
      for (int32 i = 0; i <= j; ++i)
      {
        ftmp[i] = data[cartesian_to_tri_idx(i, pmj, p+1)];
        data[cartesian_to_tri_idx(i, pmj, p+1)] = 0.0f;
      }

      // Solve 1D problem control points, resulting in c[0,...,j].
      NewtonBernstein1D_scaled(j, 1.0f/(1.0f-y[pmj]), x, w, ftmp, c);

      // Extend solution to whole triangle.
      // All control points are zero except the y=0 row,
      // so only store the y=0 row.
      const Float extend = powf(1.0f-y[pmj], -j);
      for (int32 i = 0; i <= j; ++i)
        c[i] *= extend;

      // BBProduct of order-j polynomial and the product of (p-j) linear
      // gamma polynomials, which are zero on their respective lines.
      // The gamma polynomials are functions of y only, hence their control
      // points are uniform across any row.
      // Only the control points on the x=0 edge are stored.
      // Accumulate the (triangular) product directly into the output.

      // BB 2D (triangle) multiplication formula:
      //
      // Input:  Left coefficients C_? at indices (u,v,w)  (order P)
      //         Right coefficients D_? at indices (x,y,z) (order Q)
      //
      // Output: Coefficients E_? at indices (a,b,c)       (order R=P+Q)
      //
      //     E_abc = SUM      Multichoose(P: u,v,w) * Multichoose(Q: x,y,z)   C_uvw D_xyz
      //             {u+x=a  -----------------------------------------------
      //              v+y=b            Multichoose(R: a,b,c)
      //              w+z=c}
      //
      MultinomialCoeff<2> mn_c, mn_g, mn_prod;
      mn_c.construct(j);
      mn_g.construct(pmj);
      mn_prod.construct(p);

      for (int32 di = 0; di <= pmj; ++di)   // mn_g:+1-2;  mn_prod:+1-2
      {
        for (int32 sj = 0; sj <= j; ++sj)   // mn_c:+0-2;  mn_prod:+0-2
        {
          for (int32 dj = 0; dj <= pmj-di; ++dj)  // mn_g:+0-2;  mn_prod:+0-2;
          {
            // In this loop, P=(j)    u=(sj)     v=0     w=(j-sj)           [mn_c]
            //               Q=(pmj)  x=(dj)     y=(di)  z=(pmj-di-dj)      [mn_g]
            //               R=(p)    a=(sj+dj)  b=(di)  c=(p-di-dj-sj)     [mn_prod]
            //
            assert( (sj   == mn_c.get_ijk()[0]) );
            assert( (0    == mn_c.get_ijk()[1]) );
            assert( (j-sj == mn_c.get_ijk()[2]) );
            assert( (dj        == mn_g.get_ijk()[0]) );
            assert( (di        == mn_g.get_ijk()[1]) );
            assert( (pmj-di-dj == mn_g.get_ijk()[2]) );
            assert( (sj+dj      == mn_prod.get_ijk()[0]) );
            assert( (di         == mn_prod.get_ijk()[1]) );
            assert( (p-di-dj-sj == mn_prod.get_ijk()[2]) );

            data[cartesian_to_tri_idx(sj+dj, di, p+1)] +=
              c[sj] * (gprod[di] * mn_c.get_val() * mn_g.get_val() / mn_prod.get_val());

            if (dj < pmj-di)
            {
              mn_g.slide_over(0);            // x++
              mn_prod.slide_over(0);         // a++
            }
            else
            {
              mn_g.swap_places(0, 2);                               // Reset to x=0
              for (int32 reverse = 0; reverse < pmj-di; ++reverse)  // Reset to a=sj
                mn_prod.slide_prev(0);
            }
          }//dj

          if (sj < j)
          {
            mn_c.slide_over(0);              // u++
            mn_prod.slide_over(0);           // a++
          }
          else
          {
            mn_c.swap_places(0, 2);          // Reset to u=0
            for (int32 reverse = 0; reverse < j; ++reverse)
              mn_prod.slide_prev(0);
          }
        }//sj

        if (di < pmj)
        {
          mn_g.slide_over(1);                // y++
          mn_prod.slide_over(1);             // b++
        }
      }//di
      //
      // Algorithm in paper uses successive linear triangular mutliplications,
      // but to do that here would require a secondary triangular (2D) buffer.

      // Update the product of gamma polynomials
      // by multiplying another linear factor.
      // Linear Bernstein ctrl pts such that eval== 0 @line y=y[pmj].
      const Float gamma_hi = 1.0f - y[pmj];
      const Float gamma_lo = -y[pmj];
      for (int32 i = pmj; i >= 0; --i)
      {
        gprod[i+1] += gprod[i] * gamma_hi * (Float(i+1) / Float(pmj+1));
        gprod[i] = gprod[i] * gamma_lo * (Float(pmj+1-i) / Float(pmj+1));
      }

      // Eval c @(x[diff_j], y[diff_i]):
      //   = SUM_{ej} Multichoose(j: ej, 0, j-ej) * pow(x, ej) * pow(1.0f-x-y, j-ej);

      // Eval gamma_j @ y[diff_i]
      //   = (1.0f-y[pmj])*y[diff_i] + (-y[pmj])*(1.0f-y[pmj])
      //   = y[diff_i] - y[pmj]

      // Update divided differences.
      for (int32 diff_i = pmj+1; diff_i <= p; ++diff_i)
      {
        for (int32 diff_j = 0; diff_j <= p-diff_i; ++diff_j)
        {
          data[cartesian_to_tri_idx(diff_j, diff_i, p+1)] =
            (data[cartesian_to_tri_idx(diff_j, diff_i, p+1)]
             - eops::eval_1d(OrderPolicy<General>{j}, c, x[diff_j], 1.0f-y[diff_i])) / (y[diff_i] - y[pmj]);
        }
      }
    }

    ftmp[0] = data[cartesian_to_tri_idx(0, p, p+1)];
    data[cartesian_to_tri_idx(0, p, p+1)] = 0;
    // Order-0 polynomial times order-p polynomial of y.
    for (int32 i = 0; i <= p; ++i)
      for (int32 j = 0; j <= p-i; ++j)
        data[cartesian_to_tri_idx(j, i, p+1)] += ftmp[0] * gprod[i];
  }



  /**
   * @brief Solves Bernstein interpolation problem on each element.
   *
   * TODO Handle shared degrees of freedom.
   * TODO Handle non-uniformly spaced points.
   */

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeQuad,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)
  {
    GridFunction<ncomp> out;
    out.resize(in.m_size_el, in.m_el_dofs, in.m_size_ctrl);
    array_copy(out.m_ctrl_idx, in.m_ctrl_idx);

    DeviceGridFunctionConst<ncomp> dgf_in(in);
    DeviceGridFunction<ncomp> dgf_out(out);
    const int32 nelem = in.m_size_el;
    const int32 p = p_;

    // Convert each element.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 eidx) {
        ReadDofPtr<Vec<Float, ncomp>> rdp = dgf_in.get_rdp(eidx);
        WriteDofPtr<Vec<Float, ncomp>> wdp = dgf_out.get_wdp(eidx);

        // 1D scratch space.
        //   TODO shrink this when get refined General policy.
        Vec<Float, ncomp> bF[MaxPolyOrder+1];
        Vec<Float, ncomp> bC[MaxPolyOrder+1];
        Float bW[MaxPolyOrder+1];

        // Uniform closed.  TODO more general spacing options.
        Float x[MaxPolyOrder+1];
        for (int32 i = 0; i <= p; ++i)
          x[i] = 1.0 * i / p;

        const int32 npe = (p+1)*(p+1);
        for (int32 nidx = 0; nidx < npe; ++nidx)
          wdp[nidx] = rdp[nidx];

        // i
        for (int32 j = 0; j <= p; ++j)
        {
          for (int32 i = 0; i <= p; ++i)
            bF[i] = wdp[j*(p+1) + i];
          NewtonBernstein1D(p, x, bW, bF, bC);
          for (int32 i = 0; i <= p; ++i)
            wdp[j*(p+1) + i] = bC[i];
        }

        // j
        for (int32 i = 0; i <= p; ++i)
        {
          for (int32 j = 0; j <= p; ++j)
            bF[j] = wdp[j*(p+1) + i];
          NewtonBernstein1D(p, x, bW, bF, bC);
          for (int32 j = 0; j <= p; ++j)
            wdp[j*(p+1) + i] = bC[j];
        }
    });

    return out;
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeHex,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)
  {
    GridFunction<ncomp> out;
    out.resize(in.m_size_el, in.m_el_dofs, in.m_size_ctrl);
    array_copy(out.m_ctrl_idx, in.m_ctrl_idx);

    DeviceGridFunctionConst<ncomp> dgf_in(in);
    DeviceGridFunction<ncomp> dgf_out(out);
    const int32 nelem = in.m_size_el;
    const int32 p = p_;

    // Convert each element.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 eidx) {
        ReadDofPtr<Vec<Float, ncomp>> rdp = dgf_in.get_rdp(eidx);
        WriteDofPtr<Vec<Float, ncomp>> wdp = dgf_out.get_wdp(eidx);

        // 1D scratch space.
        //   TODO shrink this when get refined General policy.
        Vec<Float, ncomp> bF[MaxPolyOrder+1];
        Vec<Float, ncomp> bC[MaxPolyOrder+1];
        Float bW[MaxPolyOrder+1];

        // Uniform closed.  TODO more general spacing options.
        Float x[MaxPolyOrder+1];
        for (int32 i = 0; i <= p; ++i)
          x[i] = 1.0 * i / p;

        const int32 npe = (p+1)*(p+1)*(p+1);
        for (int32 nidx = 0; nidx < npe; ++nidx)
          wdp[nidx] = rdp[nidx];

        // i
        for (int32 k = 0; k <= p; ++k)
          for (int32 j = 0; j <= p; ++j)
          {
            for (int32 i = 0; i <= p; ++i)
              bF[i] = wdp[k*(p+1)*(p+1) + j*(p+1) + i];
            NewtonBernstein1D(p, x, bW, bF, bC);
            for (int32 i = 0; i <= p; ++i)
              wdp[k*(p+1)*(p+1) + j*(p+1) + i] = bC[i];
          }

        // j
        for (int32 k = 0; k <= p; ++k)
          for (int32 i = 0; i <= p; ++i)
          {
            for (int32 j = 0; j <= p; ++j)
              bF[j] = wdp[k*(p+1)*(p+1) + j*(p+1) + i];
            NewtonBernstein1D(p, x, bW, bF, bC);
            for (int32 j = 0; j <= p; ++j)
              wdp[k*(p+1)*(p+1) + j*(p+1) + i] = bC[j];
          }

        // k
        for (int32 j = 0; j <= p; ++j)
          for (int32 i = 0; i <= p; ++i)
          {
            for (int32 k = 0; k <= p; ++k)
              bF[k] = wdp[k*(p+1)*(p+1) + j*(p+1) + i];
            NewtonBernstein1D(p, x, bW, bF, bC);
            for (int32 k = 0; k <= p; ++k)
              wdp[k*(p+1)*(p+1) + j*(p+1) + i] = bC[k];
          }
    });

    return out;
  }



  // Note: This function outputs disjoint elements with contiguous memory per element.
  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeTri,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)

  {
    GridFunction<ncomp> out;
    out.resize(in.m_size_el, in.m_el_dofs, in.m_size_el * in.m_el_dofs);
    out.m_ctrl_idx = array_counting(out.m_ctrl_idx.size(), 0, 1);
    // Makes wdp element memory contiguous per element.

    DeviceGridFunctionConst<ncomp> dgf_in(in);
    DeviceGridFunction<ncomp> dgf_out(out);
    const int32 nelem = in.m_size_el;
    const int32 p = p_;
    const OrderPolicy<General> order_p{p};

    // Convert each element.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 eidx) {
        ReadDofPtr<Vec<Float, ncomp>> rdp = dgf_in.get_rdp(eidx);
        WriteDofPtr<Vec<Float, ncomp>> wdp = dgf_out.get_wdp(eidx);

        // 1D scratch spaces.
        //   TODO shrink this when get refined General policy.
        Float bW[MaxPolyOrder+1];
        Float bG[MaxPolyOrder+1];
        Vec<Float, ncomp> bF[MaxPolyOrder+1];
        Vec<Float, ncomp> bC[MaxPolyOrder+1];

        // Uniform closed.  TODO more general spacing options.
        Float x[MaxPolyOrder+1];
        Float y[MaxPolyOrder+1];
        for (int32 i = 0; i <= p; ++i)
        {
          x[i] = 1.0 * i / p;
          y[i] = 1.0 * i / p;
        }

        const int32 npe = eattr::get_num_dofs(ShapeTri(), order_p);
        for (int32 nidx = 0; nidx < npe; ++nidx)
          wdp[nidx] = rdp[nidx];

        // Assumes that wdp element memory is contiguous per element.
        NewtonBernsteinTri(p, x, y, bW, bG, bF, bC, &wdp[0]);
    });

    return out;
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeTet,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)

  {
    throw std::logic_error("ToBernstein_execute(ShapeTet, gf) not implemented");
  }

  // ---------- Wrappers ------------------- //

  // ToBernsteinTopo_execute(): Get grid function and pass to ToBernstein_execute().
  template <typename MElemT>
  std::shared_ptr<Mesh> ToBernsteinTopo_execute(
      const UnstructuredMesh<MElemT> &mesh)
  {
    const GridFunction<3> &in_mesh_gf = mesh.get_dof_data();
    const GridFunction<3> out_mesh_gf =
        ToBernstein_execute(adapt_get_shape<MElemT>(), in_mesh_gf, mesh.order());
    UnstructuredMesh<MElemT> omesh(out_mesh_gf, mesh.order());

    return std::make_shared<UnstructuredMesh<MElemT>>(omesh);
  }

  // ToBernsteinField_execute(): Get grid function and pass to ToBernstein_execute().
  template <typename FElemT>
  std::shared_ptr<Field> ToBernsteinField_execute(const UnstructuredField<FElemT> &field)
  {
    constexpr int32 ncomp = FElemT::get_ncomp();
    const GridFunction<ncomp> &in_gf = field.get_dof_data();
    const GridFunction<ncomp> out_gf =
        ToBernstein_execute(adapt_get_shape<FElemT>(), in_gf, field.order());

    return std::make_shared<UnstructuredField<FElemT>>(out_gf, field.order(), field.name());
  }


  // Templated topology functor
  struct ToBernstein_TopoFunctor
  {
    std::shared_ptr<Mesh> m_output;

    template <typename MeshType>
    void operator() (MeshType &mesh)
    {
      m_output = ToBernsteinTopo_execute(mesh);
    }
  };

  // Templated field functor
  struct ToBernstein_FieldFunctor
  {
    std::shared_ptr<Field> m_output;

    template <typename FieldT>
    void operator() (FieldT &field)
    {
      m_output = ToBernsteinField_execute(field);
    }
  };

  // execute() wrapper
  DataSet ToBernstein::execute(DataSet &data_set)
  {
    ToBernstein_TopoFunctor topo_f;
    ToBernstein_FieldFunctor field_f;

    dispatch(data_set.mesh(), topo_f);
    DataSet out_ds(topo_f.m_output);

    for (const std::string &fname : data_set.fields())
    {
      dispatch(data_set.field(fname), field_f);
      out_ds.add_field(field_f.m_output);
    }

    return out_ds;
  }

  Collection ToBernstein::execute(Collection &collxn)
  {
    Collection out_collxn;
    for (DataSet ds : collxn.domains())
      out_collxn.add_domain(this->execute(ds));
    return out_collxn;
  }

}//namespace dray
