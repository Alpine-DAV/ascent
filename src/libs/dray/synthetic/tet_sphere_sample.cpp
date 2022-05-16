// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/synthetic/tet_sphere_sample.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/array_utils.hpp>
#include <dray/data_model/grid_function.hpp>
#include <dray/data_model/unstructured_mesh.hpp>
#include <dray/data_model/elem_ops.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{
  int32 index3(int32 e, int32 lvi)
  {
    return (e + lvi) % 4;
  }

  /**
   * Creates a 4-triangle surface dataset approximating the shape of a sphere.
   */
  DataSet SynthesizeTetSphereSample::synthesize_dataset() const
  {
    using MeshOrderP = OrderPolicy<-1>;
    const int32 p = this->m_params.p;
    const MeshOrderP order_p{p};
    const int32 npe = eattr::get_num_dofs(ShapeTri(), order_p);

    const Float R = this->m_params.R;

    using MElemT = Element< eattr::get_dim(ShapeTri{}),
                            3,
                            eattr::get_etype(ShapeTri{}),
                            eattr::get_policy_id(MeshOrderP{}) >;

    GridFunction<3> mesh_data;
    mesh_data.resize(4, npe, 4*npe);
    mesh_data.m_ctrl_idx = array_counting(4*npe, 0, 1);

    Vec<Float, 3> * vptr = mesh_data.m_values.get_host_ptr();

    Vec<Float, 3> verts[4] = { {{-1, -1, -1}},
                               {{ 1,  1, -1}},
                               {{ 1, -1,  1}},
                               {{-1,  1,  1}} };

    using detail::cartesian_to_tri_idx;

    for (int32 e = 0; e < 4; ++e)
    {
      Vec<Float, 3> * elem_ptr = vptr + (e*npe);
      for (int32 i = 0; i <= p; ++i)
        for (int32 j = 0; j <= p-i; ++j)
        {
          // Linear interpolation across face of triangle.
          elem_ptr[cartesian_to_tri_idx(j, i, p+1)] = verts[index3(e, 0)] * (Float(j)/p)
                                                    + verts[index3(e, 1)] * (Float(i)/p)
                                                    + verts[index3(e, 2)] * (Float(p-i-j)/p);

          // Inflate to radius approximate sphere.
          elem_ptr[cartesian_to_tri_idx(j, i, p+1)].normalize();
          elem_ptr[cartesian_to_tri_idx(j, i, p+1)] *= R;
        }
    }

    UnstructuredMesh<MElemT> mesh(mesh_data, p);
    DataSet out_dataset(std::make_shared<UnstructuredMesh<MElemT>>(mesh));

    return out_dataset;
  }

}
