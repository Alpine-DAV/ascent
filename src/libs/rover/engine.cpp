//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
  if (!m_tracer)
  {
    ROVER_ERROR("Error - Engine::validate_tracer: data was not set before tracing");
  }
}

void
Engine::set_data_set(vtkm::cont::DataSet &dataset)
{
  ROVER_INFO("Energy Engine setting data set");
  // TODO: Can we initialize the tracer in the constructor?
  // TODO: Investigate why we set an empty field name here,
  // do we delete and replace the tracer later on or do we
  // explicitly set the field names?
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
  // Return early if emission is not specified
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
  init_rays(rays);
  m_tracer->SetUnitScalar(rover::settings["rover/unit_scalar"].value());
  m_tracer->SetRenderMode(vtkm::rendering::ConnectivityProxy::RenderMode::Energy);
  m_tracer->SetColorMap(m_color_map);
  return m_tracer->PartialTrace(rays);
}

void
Engine::init_rays(Ray32 &rays)
{
  validate_tracer();
  const int num_bins = detect_num_bins();
  rays.Buffers.at(0).SetNumChannels(num_bins);
  rays.Buffers.at(0).InitConst(1.0f);
  init_emission(rays, num_bins);
}

void
Engine::init_rays(Ray64 &rays)
{
  validate_tracer();
  const int num_bins = detect_num_bins();
  rays.Buffers.at(0).SetNumChannels(num_bins);
  rays.Buffers.at(0).InitConst(1.0f);
  init_emission(rays, num_bins);
}

PartialVector64
Engine::partial_trace(Ray64 &rays)
{
  ROVER_INFO("Executing Engine::partial_trace");
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

  // TODO: Seemingly redundant assert followed by a check that num_cells == 0
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
    ROVER_ERROR("Error - Engine::detect_num_bins: absorption field size is not evenly divided by num_cells"
                << "\n       modulo " << modulo
                << "\n       num cells " << num_cells
                << "\n       field size " << absorption_size);
    throw RoverException("absorption field size is not evenly divided by num_cells\n");
  }
  vtkm::Id num_bins = absorption_size / num_cells;
  ROVER_INFO("Engine::detect_num_bins: Detected " << num_bins << " bins");
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
Engine::set_composite_background(bool on)
{
  ROVER_INFO("Executing Engine::set_composite_background");
  validate_tracer();
  m_tracer->SetCompositeBackground(on);
}

void
Engine::set_primary_range(const vtkmRange &range)
{
  ROVER_INFO("Executing Engine::set_primary_range");
  validate_tracer();
  return m_tracer->SetScalarRange(range);
}

}; //namespace rover
