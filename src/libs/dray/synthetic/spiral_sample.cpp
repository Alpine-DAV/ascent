// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/synthetic/spiral_sample.hpp>
#include <dray/policies.hpp>
#include <dray/array_utils.hpp>
#include <dray/data_model/grid_function.hpp>
#include <dray/data_model/unstructured_field.hpp>
#include <dray/data_model/unstructured_mesh.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{

  /**
   * Creates a single hex cell dataset in the shape of a spiral.
   * Reference space axes are mapped as follows:
   *   0: In-plane transverse. Rotates with spiral.
   *   1: In-plane longitudinal. Rotates with spiral.
   *   2: Out-of-plane. Always maps to world Z.
   */
  DataSet SynthesizeSpiralSample::synthesize_dataset() const
  {
    using MeshOrderP = OrderPolicy<-1>;
    const int32 p = this->m_params.p;
    const MeshOrderP order_p{p};
    const int32 npe = eattr::get_num_dofs(ShapeHex(), order_p);

    using MElemT = Element< eattr::get_dim(ShapeHex{}),
                            3,
                            eattr::get_etype(ShapeHex{}),
                            eattr::get_policy_id(MeshOrderP{}) >;

    GridFunction<3> mesh_data;
    mesh_data.resize(1, npe, npe);
    mesh_data.m_ctrl_idx = array_counting(npe, 0, 1);

    Vec<Float, 3> * vptr = mesh_data.m_values.get_device_ptr();
    const SynthesizeSpiralSample::Params pm = this->m_params;

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, npe), [=] DRAY_LAMBDA (int32 idx) {
        const int32 k = idx / ((p+1)*(p+1));
        const int32 j = (idx - k*((p+1)*(p+1))) / (p+1);
        const int32 i = (idx - k*((p+1)*(p+1)) - j*(p+1)) / 1;

        const Float H_2pi = pm.H / (2 * pi());

        const Float theta = pm.revs * 2 * pi() * j / p;
        const Float r = H_2pi * (theta + pi());
        const Float t = pm.w * (1.0 * i / p - 0.5);

        const Float c = cos(theta);
        const Float s = sin(theta);

        const Vec<Float, 2> x = {{ r*c, r*s }};
        /// const Vec<Float, 2> n = (Vec<Float, 2>{{-H_2pi*s, H_2pi*c}} - x).normalized();
        const Vec<Float, 2> n = (-x).normalized();

        Vec<Float, 3> pos = {{ x[0] - n[0]*t, x[1] - n[1]*t, (pm.w * k / p) }};

        vptr[k*((p+1)*(p+1)) + j*(p+1) + i] = pos;
    });

    UnstructuredMesh<MElemT> mesh(mesh_data, p);
    DataSet out_dataset(std::make_shared<UnstructuredMesh<MElemT>>(mesh));

    return out_dataset;
  }

}
