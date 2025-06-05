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
#include <vtkm_typedefs.hpp>

using namespace conduit;

namespace rover {

class Domain
{
public:
  Domain();
  ~Domain();
  const vtkmDataSet& get_data_set();
  PartialVector32 partial_trace(Ray32 &rays);
  PartialVector64 partial_trace(Ray64 &rays);
  void init_rays(Ray32 &rays);
  void init_rays(Ray64 &rays);
  void set_data_set(vtkmDataSet &dataset);
  void set_settings(const Node &settings);
  void set_primary_range(const vtkmRange &range);
  void set_composite_background(bool on);
  vtkm::Bounds get_domain_bounds();
  vtkmRange get_primary_range();
  void set_global_bounds(vtkm::Bounds bounds);
  int get_num_channels();
protected:
  std::shared_ptr<Engine> m_engine;
  vtkmDataSet             m_data_set;
  vtkm::Bounds            m_global_bounds;
  vtkm::Bounds            m_domain_bounds;
  // TODO: Can we either 1) make m_settings globally available within
  // the rover namespace or 2) pass the needed values from m_settings
  // as function arguments?
  Node                    m_settings;
  void                    set_engine_fields();
}; // class domain
} // namespace rover
#endif
