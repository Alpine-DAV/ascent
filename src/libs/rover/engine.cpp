//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "vtkm_typedefs.hpp"
#include <engine.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>
#include <vtkm/cont/DefaultTypes.h>

namespace rover
{

Engine::Engine()
{
  m_tracer = nullptr;
}

Engine::~Engine()
{
  if (m_tracer)
  {
    delete m_tracer;
  }
}

void
Engine::validate_tracer()
{
  // I noticed this check happens in several places,
  // so I pulled it out into a helper function.
  // However, after understanding the engine code
  // better, I suspect we'll only need to check
  // this in 1 or 2 spots and won't need a helper.
  if (!m_tracer)
  {
    ROVER_ERROR("Error - Engine::validate_tracer: data was not set before tracing");
  }
}

void
Engine::set_dataset(vtkm::cont::DataSet &dataset)
{
  ROVER_INFO("Executing Engine::set_data_set");
  // TODO: Can we initialize the tracer in the constructor?
  // TODO: Investigate why we set an empty field name here,
  // do we delete and replace the tracer later on or do we
  // explicitly set the field names?
  const std::string absorption = rover::settings["absorption"].as_string();
  m_tracer = new vtkh::rendering::ConnectivityProxy(dataset, absorption);
  m_tracer->SetScalarField(absorption);
  m_dataset = dataset;
}

template<typename Precision>
void
Engine::init_emission(vtkmRayTracing::Ray<Precision> &rays,
                      const int num_energy_groups)
{
  if (rover::settings.has_child("emission"))
  {
    const std::string emission = rover::settings["emission"].as_string();
    m_tracer->SetEmissionField(emission);
    rays.AddBuffer(num_energy_groups, "emission");
    rays.GetBuffer("emission").InitConst(0.0f);
  }
}

void
Engine::partial_trace(Ray32 &rays, PartialVector32 &partials)
{
  ROVER_INFO("Executing Engine::partial_trace");
  const bool divide_emis_by_absorb = rover::settings["divide_emis_by_absorb"].as_string() == "true";
  m_tracer->SetDivideEmisByAbsorb(divide_emis_by_absorb);
  const float64 unit_scalar = rover::settings["unit_scalar"].to_float64();
  m_tracer->SetUnitScalar(unit_scalar);
  m_tracer->PartialTrace(rays, partials);
}

void
Engine::init_rays(Ray32 &rays)
{
  validate_tracer();
  const int num_energy_groups = get_num_energy_groups();
  rays.Buffers.at(0).SetNumChannels(num_energy_groups);
  // TODO: I think this should be init with background intensities
  rays.Buffers.at(0).InitConst(1.0f);
  init_emission(rays, num_energy_groups);
  rays.AddBuffer(num_energy_groups, "optical_depths");
  rays.GetBuffer("optical_depths").InitConst(0.0f);
}

void
Engine::init_rays(Ray64 &rays)
{
  validate_tracer();
  const int num_energy_groups = get_num_energy_groups();
  rays.Buffers.at(0).SetNumChannels(num_energy_groups);
  // TODO: I think this should be init with background intensities
  rays.Buffers.at(0).InitConst(1.0f);
  init_emission(rays, num_energy_groups);
  rays.AddBuffer(num_energy_groups, "optical_depths");
  rays.GetBuffer("optical_depths").InitConst(0.0f);
}

void
Engine::partial_trace(Ray64 &rays, PartialVector64 &partials)
{
  ROVER_INFO("Executing Engine::partial_trace");
  const bool divide_emis_by_absorb = rover::settings["divide_emis_by_absorb"].as_string() == "true";
  m_tracer->SetDivideEmisByAbsorb(divide_emis_by_absorb);
  const float64 unit_scalar = rover::settings["unit_scalar"].to_float64();
  m_tracer->SetUnitScalar(unit_scalar);
  m_tracer->PartialTrace(rays, partials);
}

int
Engine::get_num_energy_groups()
{
  const std::string absorption = rover::settings["absorption"].as_string();
  const auto &field = m_dataset.GetField(absorption);
  vtkm::Id num_bins = field.GetData().GetNumberOfComponentsFlat();
  return static_cast<int>(num_bins);
}

vtkmRange
Engine::get_primary_range()
{
  ROVER_INFO("Executing Engine::get_primary_range");
  validate_tracer();
  return m_tracer->GetScalarFieldRange();
}

void
Engine::set_primary_range(const vtkmRange &range)
{
  ROVER_INFO("Executing Engine::set_primary_range");
  validate_tracer();
  m_tracer->SetScalarRange(range);
}

void
Engine::set_composite_background(bool on)
{
  ROVER_INFO("Executing Engine::set_composite_background");
  validate_tracer();
  m_tracer->SetCompositeBackground(on);
}

}; //namespace rover
