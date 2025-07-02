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

Domain::Domain(vtkmDataSet &dataset)
{
  m_engine = std::make_shared<Engine>();
  set_dataset(dataset);
}

Domain::~Domain()
{

}

//
// This should be called at the last possible moment by the
// scheduler so that the settings data sets / setting can
// be called in any order
//
void
Domain::init()
{
  //
  // Create the correct engine
  //

  ROVER_INFO("Executing Domain::init");

#if 0 // removing volume renderer
  if(m_render_settings.m_render_mode != volume &&
     settings.m_render_mode == volume)
  {
    ROVER_INFO("Render mode = volume");
    m_engine = std::make_shared<VolumeEngine>();
  }
  else if(m_render_settings.m_render_mode != energy &&
          settings.m_render_mode == energy)
  {
#endif

#if 0 // removing volume renderer
  }
  else if(m_render_settings.m_render_mode != surface &&
          settings.m_render_mode == surface)
  {
    std::cout<<"ray tracing not implemented\n";
  }
  else
  {
    //ROVER_ERROR("Unable to create the appropriate engine");
    //throw RoverException("Fatal Error: domain unable to create the apporpriate engine\n");
  }
#endif

  m_engine->set_dataset(m_dataset);

#if 0 // removing volume renderer
  if(m_render_settings.m_render_mode == volume)
  {
    ROVER_INFO("outgoing render mode = volume");
  }

  if(m_render_settings.m_render_mode == energy)
  {
    ROVER_INFO("outgoing render mode = energy");
  }
#endif

}

const int
Domain::get_num_channels()
{
  return m_engine->get_num_channels();
}

void
Domain::set_dataset(vtkmDataSet &dataset)
{
  ROVER_INFO("Setting dataset");
  m_engine->set_dataset(dataset);
  m_dataset = dataset;
  m_domain_bounds = m_dataset.GetCoordinateSystem().GetBounds();
}

const vtkmDataSet&
Domain::get_dataset()
{
  return m_dataset;
}

void
Domain::init_rays(Ray32 &rays)
{
  m_engine->init_rays(rays);
}

void
Domain::init_rays(Ray64 &rays)
{
  m_engine->init_rays(rays);
}

void
Domain::partial_trace(Ray32 &rays, PartialVector32 &partials)
{
  m_engine->partial_trace(rays, partials);
}

void
Domain::partial_trace(Ray64 &rays, PartialVector64 &partials)
{
  m_engine->partial_trace(rays, partials);
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
