// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_AFFINE_RADIAL
#define DRAY_AFFINE_RADIAL

#include <dray/types.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>

namespace dray
{

  /**
   * @brief Construct a uniform hex dataset centered at a point, equipped with ellipsoidal field.
   *
   * - If range and range_radii are positive, the result is an ellipsoid form (positive definite).
   * - If one has negative entries, the result could be a hyperboloid form (negative definite).
   *
   * (In the case that both radii and range_radii are positive and anisotropic,
   *  the result is a cube domain with a spherical, radially increasing field.)
   */
  class SynthesizeAffineRadial
  {
    public:
      struct TopoParams
      {
        Vec<int32, 3> m_extents;
        Vec<Float, 3> m_origin;
        Vec<Float, 3> m_radii;
      };
      struct FieldParams
      {
        std::string m_field_name;
        Vec<Float, 3> m_range_radii;
      };

      // Default constructor (deleted)
      SynthesizeAffineRadial() = delete;

      /**
       * Constructor
       *
       * @param extents Number of cells in each axis.
       * @param origin Point in physical space at which the dataset is centered.
       * @param radii Ellipsoidal semiaxes of the ellipsoid inscribed in the box dataset.
       */
      SynthesizeAffineRadial(const Vec<int32, 3> &extents,
                             const Vec<Float, 3> &origin,
                             const Vec<Float, 3> &radii)
        : m_topo_params{extents, origin, radii}
      { }

      /**
       * reset_topo()
       */
      SynthesizeAffineRadial & reset_topo(const Vec<int32, 3> &extents,
                                          const Vec<Float, 3> &origin,
                                          const Vec<Float, 3> &radii)
      {
        m_topo_params = {extents, origin, radii};
        return *this;
      }

      /**
       * equip()
       *
       * @param field_name Name of the field to add.
       * @param range_radii Values of field at midpoints of the sides of the box dataset.
       */
      SynthesizeAffineRadial & equip(const std::string &field_name,
                                     const Vec<Float, 3> &range_radii)
      {
        m_field_params_list.emplace_back(FieldParams{field_name, range_radii});
        return *this;
      }


      const TopoParams & topo_params()
      {
        return m_topo_params;
      }

      const std::vector<FieldParams> & field_params_list()
      {
        return m_field_params_list;
      }


      /**
       * synthesize()
       *
       * @brief Return a new dataset with the attributes set so far.
       */
      DataSet synthesize_dataset();
      Collection synthesize() //TODO only one mpi rank should synthesize.
      {
        Collection col;
        col.add_domain(this->synthesize_dataset());
        return col;
      }


    protected:
      TopoParams m_topo_params;
      std::vector<FieldParams> m_field_params_list;
  };

}//namespace dray

#endif
