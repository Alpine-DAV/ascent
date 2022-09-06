// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SPIRAL_SAMPLE
#define DRAY_SPIRAL_SAMPLE

#include <dray/types.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>

namespace dray
{
  /**
   * Creates a single hex cell dataset in the shape of a spiral.
   * Reference space axes are mapped as follows:
   *   0: In-plane transverse. Rotates with spiral.
   *   1: In-plane longitudinal. Rotates with spiral.
   *   2: Out-of-plane. Always maps to world Z.
   *
   * @param H Radial increase per revolution.
   * @param w Side length of square cross-section.
   * @param revs Number of revolutions from one end of the hex to the other.
   * @param p Polynomial order of the element, e.g. 20.
   */
  class SynthesizeSpiralSample
  {
    public:
      struct Params
      {
        Float H;
        Float w;
        Float revs;
        int32 p;
      };

      SynthesizeSpiralSample() = delete;

      SynthesizeSpiralSample(Float H, Float w, Float revs, int32 p)
        : m_params{H, w, revs, p}
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
