//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "viskores_typedefs.hpp"
#include <engine.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>
#include <ascent_logging.hpp>
#include <viskores/cont/DefaultTypes.h>

namespace rover
{

Engine::Engine()
{
  m_tracer = nullptr;
  m_field_mismatch_error = false;
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
    ASCENT_LOG_ERROR("Error - Engine::validate_tracer: data was not set before tracing");
  }
}

void
Engine::set_dataset(viskores::cont::DataSet &dataset)
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
<<<<<<< HEAD
Engine::init_emission(vtkmRayTracing::Ray<Precision> &rays,
                      const int num_energy_groups)
=======
Engine::init_emission(viskoresRayTracing::Ray<Precision> &rays,
                      const int num_bins)
>>>>>>> task/9_18_25-1607-move-from-vtk-m-to-viskores
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
<<<<<<< HEAD
  const std::string absorption = rover::settings["absorption"].as_string();
  const vtkm::cont::Field &absorption_field = m_dataset.GetField(absorption);
  vtkm::Id num_absorption_bins = absorption_field.GetData().GetNumberOfComponentsFlat();

  // If the emission field is set, verify that it has the same number of energy groups
  // as the absorption field
  if (rover::settings.has_child("emission"))
=======
  viskores::Id absorption_size = 0;
  ArraySizeFunctor functor(&absorption_size);
  const std::string absorption = rover::settings["absorption"].as_string();
  m_dataset.GetField(absorption).
                     GetData().
                     CastAndCallForTypes<viskores::TypeListAll, VISKORES_DEFAULT_STORAGE_LIST>(functor);
  viskores::Id num_cells = m_dataset.GetCellSet().GetNumberOfCells();

  // TODO: Seemingly redundant assert followed by a check that num_cells == 0
  assert(num_cells > 0);
  assert(absorption_size > 0);
  if (num_cells == 0)
  {
    ROVER_ERROR("Error - Engine::get_num_channels: num cells is 0"
                << "\n        num cells " << num_cells
                << "\n        field size " <<a bsorption_size);
    m_dataset.PrintSummary(std::cerr);
    throw RoverException("Failed to detect bins. Num cells cannot be 0\n");
  }

  viskores::Id modulo = absorption_size % num_cells;
  if (modulo != 0)
>>>>>>> task/9_18_25-1607-move-from-vtk-m-to-viskores
  {
    const std::string emission = rover::settings["emission"].as_string();
    const vtkm::cont::Field &emission_field = m_dataset.GetField(emission);
    vtkm::Id num_emission_bins = emission_field.GetData().GetNumberOfComponentsFlat();

    if (num_absorption_bins != num_emission_bins)
    {
      m_field_mismatch_error = true;
    }
  }
<<<<<<< HEAD

  return static_cast<int>(num_absorption_bins);
}

bool
Engine::get_field_mismatch_error()
{
  return m_field_mismatch_error;
=======
  viskores::Id num_bins = absorption_size / num_cells;
  ROVER_INFO("Engine::get_num_channels: Detected " << num_bins << " bins");
  return static_cast<int>(num_bins);
>>>>>>> task/9_18_25-1607-move-from-vtk-m-to-viskores
}

viskoresRange
Engine::get_primary_range()
{
  ROVER_INFO("Executing Engine::get_primary_range");
  validate_tracer();
  return m_tracer->GetScalarFieldRange();
}

void
Engine::set_primary_range(const viskoresRange &range)
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
