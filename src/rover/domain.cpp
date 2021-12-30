//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <domain.hpp>
#include <volume_engine.hpp>
#include <energy_engine.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>

namespace rover {
Domain::Domain()
{
  m_engine = std::make_shared<VolumeEngine>();
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
Domain::set_render_settings(const RenderSettings &settings)
{
  //
  // Create the correct engine
  //

  ROVER_INFO("Setting render settings");

  if(m_render_settings.m_render_mode != volume &&
     settings.m_render_mode == volume)
  {
    ROVER_INFO("Render mode = volume");
    m_engine = std::make_shared<VolumeEngine>();
  }
  else if(m_render_settings.m_render_mode != energy &&
          settings.m_render_mode == energy)
  {
    ROVER_INFO("Render mode = energy");
    auto engine = std::make_shared<EnergyEngine>();
    engine->set_unit_scalar(settings.m_energy_settings.m_unit_scalar);
    m_engine = engine;
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

  m_render_settings = settings;
  m_render_settings.print();

  m_engine->set_data_set(m_data_set);
  set_engine_fields();

  if(m_render_settings.m_render_mode == volume)
  {
    ROVER_INFO("outgoing render mode = volume");
  }

  if(m_render_settings.m_render_mode == energy)
  {
    ROVER_INFO("outgoing render mode = energy");
  }
}

int
Domain::get_num_channels()
{
  return m_engine->get_num_channels();
}

void
Domain::set_data_set(vtkmDataSet &dataset)
{
  ROVER_INFO("Setting dataset");
  m_engine->set_data_set(dataset);
  m_data_set = dataset;
  m_domain_bounds = m_data_set.GetCoordinateSystem().GetBounds();
}

void
Domain::set_engine_fields()
{
  ROVER_INFO("Primary field: " << m_render_settings.m_primary_field);
  ROVER_INFO("Secondary field: " << m_render_settings.m_secondary_field);

  if(m_render_settings.m_primary_field == "")
    throw RoverException("Fatal Error: primary field not set\n");
  m_engine->set_primary_field(m_render_settings.m_primary_field);
  m_engine->set_secondary_field(m_render_settings.m_secondary_field);
  m_engine->set_color_table(m_render_settings.m_color_table);
}

const vtkmDataSet&
Domain::get_data_set()
{
  return m_data_set;
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

PartialVector32
Domain::partial_trace(Ray32 &rays)
{
  m_engine->set_samples(m_global_bounds,
                        m_render_settings.m_volume_settings.m_num_samples);
  return m_engine->partial_trace(rays);
}

PartialVector64
Domain::partial_trace(Ray64 &rays)
{
  m_engine->set_samples(m_global_bounds,
                        m_render_settings.m_volume_settings.m_num_samples);
  return m_engine->partial_trace(rays);
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
  assert(m_render_settings.m_primary_field != "");
  return m_engine->get_primary_range();
}

vtkm::Bounds
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
