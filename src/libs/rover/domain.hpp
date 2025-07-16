//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_domain_h
#define rover_domain_h

#include <rover_config.h>

#include <memory>

#include <conduit.hpp>
#include <engine.hpp>
#include <vector>
#include <vtkm_typedefs.hpp>
#include <settings.hpp>

using namespace conduit;

namespace rover {

class Domain
{
public:
  Domain();
  Domain(const vtkmDataSet &dataset);
  ~Domain();

  void set_dataset(const vtkmDataSet &dataset);

  void partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                     std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float32>> &partials,
                     const vtkm::cont::ArrayHandle<vtkm::Float32> &background);

  void partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float64> &rays,
                      std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float64>> &partials,
                      const vtkm::cont::ArrayHandle<vtkm::Float64> &background);

  const vtkmDataSet& get_dataset();
  void set_num_channels(const int num_channels);
  void set_primary_range(const vtkmRange &range);
  void set_composite_background(bool on);
  vtkm::Bounds& get_domain_bounds();
  vtkmRange get_primary_range();
  void set_global_bounds(vtkm::Bounds bounds);

protected:
  std::shared_ptr<Engine> m_engine;
  vtkmDataSet             m_dataset;
  vtkm::Bounds            m_global_bounds;
  vtkm::Bounds            m_domain_bounds;
}; // class domain
} // namespace rover
#endif
