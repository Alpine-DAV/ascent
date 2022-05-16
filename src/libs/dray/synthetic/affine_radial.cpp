// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/synthetic/affine_radial.hpp>

#include <dray/data_model/elem_attr.hpp>
#include <dray/data_model/element.hpp>

#include <dray/data_model/grid_function.hpp>
#include <dray/data_model/field.hpp>
#include <dray/data_model/mesh.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/data_model/unstructured_mesh.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>

namespace dray
{

  /**
   * synthesize()
   *
   * @brief Return a new dataset with the attributes set so far.
   */
  DataSet SynthesizeAffineRadial::synthesize_dataset()
  {
    // Mesh:     x = (2*(i+u)/n - 1) * radius_x + origin_x;
    //           y = (2*(j+v)/n - 1) * radius_y + origin_y;
    //           z = (2*(k+w)/n - 1) * radius_z + origin_z;
    //
    // Field:    f =   ((x-origin_x)/radius_x)^2 * range_radius_x
    //               + ((y-origin_y)/radius_y)^2 * range_radius_y
    //               + ((z-origin_z)/radius_z)^2 * range_radius_z
    //             (quadratic)
    //
    // Bernstein:
    //   beta0(u) = (1-u)^2;    beta1(u) = (1-u)u;    beta2(u) = u^2
    //
    //   ==>
    //           fX =   (2i/n - 1)^2 * beta0(u)
    //                + (2i/n - 1)(2(i+1)/n - 1) * beta1(u)
    //                + (2(i+1)/n - 1)^2 * beta2(u);
    //           fY =   (2j/n - 1)^2 * beta0(v)
    //                + (2j/n - 1)(2(j+1)/n - 1) * beta1(v)
    //                + (2(j+1)/n - 1)^2 * beta2(v);
    //           fZ =   (2k/n - 1)^2 * beta0(w)
    //                + (2k/n - 1)(2(k+1)/n - 1) * beta1(w)
    //                + (2(k+1)/n - 1)^2 * beta2(w);
    //
    //  Direct sum
    //
    //    f = fX + fY + fZ
    //
    //      =   (fX + fY + (2k/n - 1)^2) * beta0(w)
    //        + (fX + fY + (2k/n - 1)(2(k+1)/n - 1)) * beta1(w)
    //        + (fX + fY + (2(k+1)/n - 1)^2 * beta2(w)
    //
    //      etc.

    // Use ShapeHex and OrderPolicy<1> & OrderPolicy<2>

    using MeshOrderP = OrderPolicy<1>;
    using FieldOrderP = OrderPolicy<2>;

    constexpr int32 space_dim = 3;
    constexpr int32 field_dim = 1;

    // while Element exists
    using MElemT = Element< eattr::get_dim(ShapeHex{}),
                            space_dim,
                            eattr::get_etype(ShapeHex{}),
                            eattr::get_policy_id(MeshOrderP{}) >;

    using FElemT = Element< eattr::get_dim(ShapeHex{}),
                            field_dim,
                            eattr::get_etype(ShapeHex{}),
                            eattr::get_policy_id(FieldOrderP{}) >;

    // TODO portable way to create KernelPolicy,
    // because repeating RAJA::cuda_exec<> does not work.
    //for_cpu_policy
    using KJI_EXECPOL = RAJA::KernelPolicy<
                          RAJA::statement::For<2, for_cpu_policy,
                            RAJA::statement::For<1, for_cpu_policy,
                              RAJA::statement::For<0, for_cpu_policy,
                                RAJA::statement::Lambda<0>
                        > > > >;
    RAJA_INDEX_VALUE(KIDX, "KIDX");
    RAJA_INDEX_VALUE(JIDX, "JIDX");
    RAJA_INDEX_VALUE(IIDX, "IIDX");

    const Vec<int32, 3> ex = m_topo_params.m_extents;
    const Vec<Float, 3> orig = m_topo_params.m_origin;
    const Vec<Float, 3> radii = m_topo_params.m_radii;

    //
    // Mesh.
    //
    GridFunction<space_dim> mesh_data;
    mesh_data.resize(ex[0]*ex[1]*ex[2], 8, (ex[0]+1)*(ex[1]+1)*(ex[2]+1));
    int32 * mesh_ctrl_idx_ptr = mesh_data.m_ctrl_idx.get_host_ptr();
    Vec<Float, space_dim> * mesh_values_ptr = mesh_data.m_values.get_host_ptr();

    // Initialize the mesh values.
    RAJA::kernel<KJI_EXECPOL>(RAJA::make_tuple( RAJA::TypedRangeSegment<IIDX>(0, ex[0]+1),
                                                RAJA::TypedRangeSegment<JIDX>(0, ex[1]+1),
                                                RAJA::TypedRangeSegment<KIDX>(0, ex[2]+1) ),
    [=] /*DRAY_LAMBDA*/ (IIDX i_, JIDX j_, KIDX k_)
    {
      const int32 i = *i_, j = *j_, k = *k_;
      Vec<Float, 3> vertex = {{ 2.0f*i/ex[0] - 1.0f,  2.0f*j/ex[1] - 1.0f,  2.0f*k/ex[2] - 1.0f }};
      vertex[0] *= radii[0];
      vertex[1] *= radii[1];
      vertex[2] *= radii[2];
      vertex += orig;

      mesh_values_ptr[ k * (ex[1]+1)*(ex[0]+1) + j * (ex[0]+1) + i ] = vertex;
    });

    // Initialize the mesh dof map (ctrl_idx).
    RAJA::kernel<KJI_EXECPOL>(RAJA::make_tuple( RAJA::TypedRangeSegment<IIDX>(0, ex[0]),
                                                RAJA::TypedRangeSegment<JIDX>(0, ex[1]),
                                                RAJA::TypedRangeSegment<KIDX>(0, ex[1]) ),
    [=] /*DRAY_LAMBDA*/ (IIDX i_, JIDX j_, KIDX k_)
    {
      const int32 i = *i_, j = *j_, k = *k_;
      const int32 eidx = k * ex[1]*ex[0] + j * ex[0] + i;
      mesh_ctrl_idx_ptr[8*eidx + 0] = ((k+0) * (ex[1]+1)*(ex[0]+1) + (j+0) * (ex[0]+1) + (i+0));
      mesh_ctrl_idx_ptr[8*eidx + 1] = ((k+0) * (ex[1]+1)*(ex[0]+1) + (j+0) * (ex[0]+1) + (i+1));
      mesh_ctrl_idx_ptr[8*eidx + 2] = ((k+0) * (ex[1]+1)*(ex[0]+1) + (j+1) * (ex[0]+1) + (i+0));
      mesh_ctrl_idx_ptr[8*eidx + 3] = ((k+0) * (ex[1]+1)*(ex[0]+1) + (j+1) * (ex[0]+1) + (i+1));
      mesh_ctrl_idx_ptr[8*eidx + 4] = ((k+1) * (ex[1]+1)*(ex[0]+1) + (j+0) * (ex[0]+1) + (i+0));
      mesh_ctrl_idx_ptr[8*eidx + 5] = ((k+1) * (ex[1]+1)*(ex[0]+1) + (j+0) * (ex[0]+1) + (i+1));
      mesh_ctrl_idx_ptr[8*eidx + 6] = ((k+1) * (ex[1]+1)*(ex[0]+1) + (j+1) * (ex[0]+1) + (i+0));
      mesh_ctrl_idx_ptr[8*eidx + 7] = ((k+1) * (ex[1]+1)*(ex[0]+1) + (j+1) * (ex[0]+1) + (i+1));
    });

    UnstructuredMesh<MElemT> mesh(mesh_data, 1);
    DataSet out_dataset(std::make_shared<UnstructuredMesh<MElemT>>(mesh));


    //
    // Fields.
    //
    for (const FieldParams fparams : m_field_params_list)
    {
      const Vec<Float, 3> fradii = fparams.m_range_radii;

      GridFunction<field_dim> field_data;
      field_data.resize(ex[0]*ex[1]*ex[2], 27, (2*ex[0]+1)*(2*ex[1]+1)*(2*ex[2]+1));
      int32 * field_ctrl_idx_ptr = field_data.m_ctrl_idx.get_host_ptr();
      Vec<Float, field_dim> * field_values_ptr = field_data.m_values.get_host_ptr();

      // Initialize the field values.
      RAJA::kernel<KJI_EXECPOL>(RAJA::make_tuple( RAJA::TypedRangeSegment<IIDX>(0, 2*ex[0]+1),
                                                  RAJA::TypedRangeSegment<JIDX>(0, 2*ex[1]+1),
                                                  RAJA::TypedRangeSegment<KIDX>(0, 2*ex[2]+1) ),
      [=] /*DRAY_LAMBDA*/ (IIDX i_, JIDX j_, KIDX k_)
      {
        const int32 i = *i_, j = *j_, k = *k_;

        // Even-->vertex
        // Odd--->edge

        // The symmetric tri-quadratic polynomial looks like a tensor sum.
        Float val = 0.0f;
        { const int32 ei = (i >> 1);
          const bool odd = (ei << 1) != i;
          const Float M0 = (2.0f * (ei      ) / ex[0] - 1.0f);
          const Float M1 = (2.0f * (ei + odd) / ex[0] - 1.0f);
          val += M0 * M1 * fradii[0];
        }
        { const int32 ej = (j >> 1);
          const bool odd = (ej << 1) != j;
          const Float M0 = (2.0f * (ej      ) / ex[1] - 1.0f);
          const Float M1 = (2.0f * (ej + odd) / ex[1] - 1.0f);
          val += M0 * M1 * fradii[1];
        }
        { const int32 ek = (k >> 1);
          const bool odd = (ek << 1) != k;
          const Float M0 = (2.0f * (ek      ) / ex[2] - 1.0f);
          const Float M1 = (2.0f * (ek + odd) / ex[2] - 1.0f);
          val += M0 * M1 * fradii[2];
        }

        field_values_ptr[ k * (2*ex[1]+1)*(2*ex[0]+1) + j * (2*ex[0]+1) + i ] = val;
      });

      // Initialize the field dof map (ctrl_idx).
      RAJA::kernel<KJI_EXECPOL>(RAJA::make_tuple( RAJA::TypedRangeSegment<IIDX>(0, ex[0]),
                                                  RAJA::TypedRangeSegment<JIDX>(0, ex[1]),
                                                  RAJA::TypedRangeSegment<KIDX>(0, ex[2]) ),
      [=] /*DRAY_LAMBDA*/ (IIDX i_, JIDX j_, KIDX k_)
      {
        const int32 i = *i_, j = *j_, k = *k_;

        const int32 eidx = k * ex[1]*ex[0] + j * ex[0] + i;
        const int32 k_offset = (2*ex[1]+1)*(2*ex[0]+1);
        const int32 j_offset = (2*ex[0]+1);
        const int32 j0 = (2*j+0) * j_offset;
        const int32 j1 = (2*j+1) * j_offset;
        const int32 j2 = (2*j+2) * j_offset;
        const int32 k0 = (2*k+0) * k_offset;
        const int32 k1 = (2*k+1) * k_offset;
        const int32 k2 = (2*k+2) * k_offset;

        field_ctrl_idx_ptr[27*eidx +  0] = k0 + j0 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx +  1] = k0 + j0 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx +  2] = k0 + j0 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx +  3] = k0 + j1 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx +  4] = k0 + j1 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx +  5] = k0 + j1 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx +  6] = k0 + j2 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx +  7] = k0 + j2 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx +  8] = k0 + j2 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx +  9] = k1 + j0 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx + 10] = k1 + j0 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx + 11] = k1 + j0 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx + 12] = k1 + j1 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx + 13] = k1 + j1 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx + 14] = k1 + j1 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx + 15] = k1 + j2 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx + 16] = k1 + j2 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx + 17] = k1 + j2 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx + 18] = k2 + j0 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx + 19] = k2 + j0 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx + 20] = k2 + j0 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx + 21] = k2 + j1 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx + 22] = k2 + j1 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx + 23] = k2 + j1 + (2*i+2);
        field_ctrl_idx_ptr[27*eidx + 24] = k2 + j2 + (2*i+0);
        field_ctrl_idx_ptr[27*eidx + 25] = k2 + j2 + (2*i+1);
        field_ctrl_idx_ptr[27*eidx + 26] = k2 + j2 + (2*i+2);
      });

      std::shared_ptr<UnstructuredField<FElemT>> field
        = std::make_shared<UnstructuredField<FElemT>>(field_data, 2);
      field->name(fparams.m_field_name);

      out_dataset.add_field(field);
    }

    return out_dataset;
  }

}//namespace dray
