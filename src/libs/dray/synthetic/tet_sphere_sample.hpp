// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TET_SPHERE_SAMPLE
#define DRAY_TET_SPHERE_SAMPLE

#include <dray/types.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>

namespace dray
{
  /**
   * Creates a 4-triangle surface dataset approximating the shape of a sphere.
   *
   * @param R Sphere radius.
   * @param p Polynomial order of the elements, e.g. 4.
   */
  class SynthesizeTetSphereSample
  {
    public:
      struct Params
      {
        Float R;
        int32 p;
      };

      SynthesizeTetSphereSample() = delete;

      SynthesizeTetSphereSample(Float R, int32 p)
        : m_params{R, p}
      {}

      DataSet synthesize_dataset() const;
      Collection synthesize() const //TODO only one mpi rank should synthesize.
      {
        Collection col;
        col.add_domain(this->synthesize_dataset());
        return col;
      }

    protected:
      Params m_params;
  };
}

#endif
