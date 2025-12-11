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
#include <viskores_typedefs.hpp>
#include <settings.hpp>

using namespace conduit;

namespace rover {

class Domain
{
public:
  Domain();
  Domain(viskoresDataSet &dataset);
  ~Domain();

  void init();
  const viskoresDataSet& get_dataset();
  void partial_trace(Ray32 &rays, PartialVector32 &partials);
  void partial_trace(Ray64 &rays, PartialVector64 &partials);
  void init_rays(Ray32 &rays);
  void init_rays(Ray64 &rays);
  void set_dataset(viskoresDataSet &dataset);
  void set_primary_range(const viskoresRange &range);
  void set_composite_background(bool on);
  viskores::Bounds& get_domain_bounds();
  viskoresRange get_primary_range();
  void set_global_bounds(viskores::Bounds bounds);
  const int get_num_energy_groups();
  bool get_field_mismatch_error();
protected:
  std::shared_ptr<Engine> m_engine;
  viskoresDataSet             m_dataset;
  viskores::Bounds            m_global_bounds;
  viskores::Bounds            m_domain_bounds;
}; // class domain
} // namespace rover
#endif
