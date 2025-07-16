//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "vtkm_typedefs.hpp"
#include <domain.hpp>
#include <engine.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>

namespace rover
{

Domain::Domain()
{
  m_engine = std::make_shared<Engine>();
}

Domain::Domain(const vtkmDataSet &dataset)
{
  m_engine = std::make_shared<Engine>();
  set_dataset(dataset);
}

Domain::~Domain()
{

}

void
Domain::set_dataset(const vtkmDataSet &dataset)
{
  ROVER_INFO("Setting dataset");
  m_engine->set_dataset(dataset);
  m_dataset = dataset;
  m_domain_bounds = m_dataset.GetCoordinateSystem().GetBounds();
}

void
Domain::set_num_channels(const int num_channels)
{
  m_engine->set_num_channels(num_channels);
}

void
Domain::partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                      std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float32>> &partials,
                      const vtkm::cont::ArrayHandle<vtkm::Float32> &background)
{
  m_engine->partial_trace(rays, partials, background);
}

void
Domain::partial_trace(vtkm::rendering::raytracing::Ray<vtkm::Float64> &rays,
                      std::vector<vtkh::rendering::raytracing::PartialComposite<vtkm::Float64>> &partials,
                      const vtkm::cont::ArrayHandle<vtkm::Float64> &background)
{
  m_engine->partial_trace(rays, partials, background);
}

const vtkmDataSet&
Domain::get_dataset()
{
  return m_dataset;
}

void
Domain::set_primary_range(const vtkmRange &range)
{
  m_engine->set_primary_range(range);
}

void
Domain::set_composite_background(bool on)
{
  m_engine->set_composite_background(on);
}

vtkmRange
Domain::get_primary_range()
{
  return m_engine->get_primary_range();
}

vtkm::Bounds&
Domain::get_domain_bounds()
{
  return m_domain_bounds;
}

void
Domain::set_global_bounds(vtkm::Bounds bounds)
{
  m_global_bounds = bounds;
}

} // namespace rover
