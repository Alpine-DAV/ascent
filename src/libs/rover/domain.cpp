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

namespace rover
{

Domain::Domain()
{
  m_engine = std::make_shared<EnergyEngine>();
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

  auto engine = std::make_shared<EnergyEngine>();
  engine->set_unit_scalar(rover::settings["rover/unit_scalar"].value());
  m_engine = engine;

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

  m_engine->set_data_set(m_data_set);
  set_engine_fields();

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
  const std::string absorption = rover::settings["rover/absorption"].as_string();
  const std::string emission = rover::settings["rover/emission"].as_string();
  const std::string color_table_name = rover::settings["rover/color_table"].as_string();
  vtkmColorTable color_table(color_table_name);

  ROVER_INFO("Primary (absorption) field: " << absorbtion);
  ROVER_INFO("Secondary (emission) field: " << emission);

  m_engine->set_primary_field(absorption);
  m_engine->set_secondary_field(emission);
  m_engine->set_color_table(color_table);
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
  int32 num_samples = rover::settings["rover/num_samples"].value();
  m_engine->set_samples(m_global_bounds, num_samples);
  return m_engine->partial_trace(rays);
}

PartialVector64
Domain::partial_trace(Ray64 &rays)
{
  int32 num_samples = rover::settings["rover/num_samples"].value();
  m_engine->set_samples(m_global_bounds, num_samples);
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
