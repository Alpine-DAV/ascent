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
  m_num_channels = 1;
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
Engine::set_dataset(const vtkm::cont::DataSet &dataset)
{
  ROVER_INFO("Executing Engine::set_dataset");
  // TODO: Can we initialize the tracer in the constructor?
  const std::string absorption = rover::settings["absorption"].as_string();
  m_tracer = new vtkh::rendering::ConnectivityProxy(dataset, absorption);

  const std::string emission = rover::settings["emission"].as_string();
  if (!emission.empty())
  {
    m_tracer->SetEmissionField(emission);
  }

  const bool divide_emis_by_absorb = rover::settings["divide_emis_by_absorb"].as_string() == "true";
  m_tracer->SetDivideEmisByAbsorb(divide_emis_by_absorb);

  const float64 unit_scalar = rover::settings["unit_scalar"].to_float64();
  m_tracer->SetUnitScalar(unit_scalar);
}

void
Engine::set_num_channels(const int num_channels)
{
  ROVER_INFO("Executing Engine::set_num_channels");
  m_num_channels = num_channels;
}

void
Engine::partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                      std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float32>> &partials,
                      const vtkm::cont::ArrayHandle<vtkm::Float32> &background)
{
  ROVER_INFO("Executing Engine::partial_trace");
  validate_tracer();
  
  // Init the ray buffers
  rays.AddBuffer(m_num_channels, "intensity");
  rays.GetBuffer("intensity").InitChannels(background);
  rays.AddBuffer(m_num_channels, "optical_depth");
  rays.GetBuffer("optical_depth").InitConst(0.0f);
  m_tracer->PartialTrace(rays, partials);
}

void
Engine::partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float64> &rays,
                      std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float64>> &partials,
                      const vtkm::cont::ArrayHandle<vtkm::Float64> &background)
{
  ROVER_INFO("Executing Engine::partial_trace");
  validate_tracer();
  
  // Init the ray buffers
  rays.AddBuffer(m_num_channels, "intensity");
  rays.GetBuffer("intensity").InitChannels(background);
  rays.AddBuffer(m_num_channels, "optical_depth");
  rays.GetBuffer("optical_depth").InitConst(0.0f);
  m_tracer->PartialTrace(rays, partials);
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
