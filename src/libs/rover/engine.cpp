//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "settings.hpp"
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
Engine::set_data_set(vtkm::cont::DataSet &dataset)
{
  ROVER_INFO("Energy Engine settting data set");
  if(m_tracer) delete m_tracer;

  m_tracer = new vtkm::rendering::ConnectivityProxy(dataset, "");
  m_tracer->SetRenderMode(vtkm::rendering::ConnectivityProxy::RenderMode::Energy);
  m_data_set = dataset;

}

int
Engine::get_num_channels()
{
  return detect_num_bins();
}

void
Engine::set_primary_field()
{
  const std::string absorption = rover::settings["rover/absorption"].as_string();
  ROVER_INFO("Engine::set_primary_field: using '" << absorption << "'");
  m_tracer->SetScalarField(absorption);
}

void
Engine::set_secondary_field()
{
  std::string emission = rover::settings["rover/emission"].as_string();
  // Return early if emission was not specified
  if("" == emission)
  {
    ROVER_INFO("Engine::set_secondary_field: emission not specified");
    return;
  }

  ROVER_INFO("Engine::set_secondary_field: using '" << emission << "'");
  m_tracer->SetEmissionField(emission);
}

template<typename Precision>
void
Engine::init_emission(vtkm::rendering::raytracing::Ray<Precision> &rays,
                            const int num_bins)
{
  const std::string emission = rover::settings["rover/emission"].as_string();
  // Return early if emission was not specified
  if("" == emission)
  {
    return;
  }

  rays.AddBuffer(num_bins, "emission");
  rays.GetBuffer("emission").InitConst(0);
}

PartialVector32
Engine::partial_trace(Ray32 &rays)
{
  ROVER_INFO("Executing Engine::partial_trace");
  if (!m_tracer)
  {
    ROVER_ERROR("Error - Engine::partial_trace: data was not set before tracing");
  }

  init_rays(rays);
  m_tracer->SetUnitScalar(rover::settings["rover/unit_scalar"].value());
  m_tracer->SetRenderMode(vtkm::rendering::ConnectivityProxy::RenderMode::Energy);
  m_tracer->SetColorMap(m_color_map);
  return m_tracer->PartialTrace(rays);
}

void
Engine::init_rays(Ray32 &rays)
{
  int num_bins = detect_num_bins();
  rays.Buffers.at(0).SetNumChannels(num_bins);
  rays.Buffers.at(0).InitConst(1.);
  init_emission(rays, num_bins);
}

void
Engine::init_rays(Ray64 &rays)
{
  int num_bins = detect_num_bins();
  rays.Buffers.at(0).SetNumChannels(num_bins);
  rays.Buffers.at(0).InitConst(1.);
  init_emission(rays, num_bins);
}

PartialVector64
Engine::partial_trace(Ray64 &rays)
{
  ROVER_INFO("Executing Engine::partial_trace");
  if (!m_tracer)
  {
    ROVER_ERROR("Error - Engine::partial_trace: data was not set before tracing");
  }

  init_rays(rays);
  m_tracer->SetUnitScalar(rover::settings["rover/unit_scalar"].value());
  m_tracer->SetRenderMode(vtkm::rendering::ConnectivityProxy::RenderMode::Energy);
  m_tracer->SetColorMap(m_color_map);
  return m_tracer->PartialTrace(rays);
}

int
Engine::detect_num_bins()
{
  vtkm::Id absorption_size = 0;
  ArraySizeFunctor functor(&absorption_size);
  const std::string absorption = rover::settings["rover/absorption"].as_string();
  m_data_set.GetField(absorption).
                      GetData().
                      CastAndCallForTypes<vtkm::TypeListAll, VTKM_DEFAULT_STORAGE_LIST>(functor);
  vtkm::Id num_cells = m_data_set.GetCellSet().GetNumberOfCells();

  // TODO: Seemingly redundant assert + immediate conditional check
  assert(num_cells > 0);
  assert(absorption_size > 0);
  if (num_cells == 0)
  {
    ROVER_ERROR("Error - Engine::detect_num_bins: num cells is 0"
                << "\n        num cells " << num_cells
                << "\n        field size " <<a bsorption_size);
    m_data_set.PrintSummary(std::cerr);
    throw RoverException("Failed to detect bins. Num cells cannot be 0\n");
  }

  vtkm::Id modulo = absorption_size % num_cells;
  if (modulo != 0)
  {
    ROVER_ERROR("Error - Engine::detect_num_bins: absoption does not evenly divide the number of cells"
                << "\n       modulo " << modulo
                << "\n       num cells " << num_cells
                << "\n       field size " << absorption_size);
    throw RoverException("absorption field size invalid (Is not evenly divided by number of cells\n");
  }
  vtkm::Id num_bins = absorption_size / num_cells;
  ROVER_INFO("Detected " << num_bins << " bins");
  return static_cast<int>(num_bins);
}

vtkmRange
Engine::get_primary_range()
{
  ROVER_INFO("Executing Engine::get_primary_range");
  if (!m_tracer)
  {
    ROVER_ERROR("Error - Engine::get_primary_range: data was not set before tracing");
  }
  return m_tracer->GetScalarFieldRange();
}

void
Engine::set_composite_background(bool on)
{
  ROVER_INFO("Executing Engine::set_composite_background");
  if (!m_tracer)
  {
    ROVER_ERROR("Error - Engine::set_composite_background: data was not set before tracing");
  }
  m_tracer->SetCompositeBackground(on);
};

void
Engine::set_primary_range(const vtkmRange &range)
{
  ROVER_INFO("Executing Engine::set_primary_range");
  if (!m_tracer)
  {
    ROVER_ERROR("Error - Engine::set_primary_range: data was not set before tracing");
  }
  return m_tracer->SetScalarRange(range);
}

}; //namespace rover
