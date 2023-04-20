//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <scheduler_base.hpp>
#include <utils/rover_logging.hpp>
namespace rover {

SchedulerBase::SchedulerBase()
{
}

SchedulerBase::~SchedulerBase()
{

}

void
SchedulerBase::set_render_settings(const RenderSettings render_settings)
{
  //
  //  In the serial schedular, the only setting that matter are
  //  m_render_mode and m_scattering_mode
  //
  m_render_settings = render_settings;
}

void
SchedulerBase::set_ray_generator(RayGenerator *ray_generator)
{
  m_ray_generator = ray_generator;
}

void
SchedulerBase::set_background(const std::vector<vtkm::Float64> &background)
{
  m_background = background;
}

void
SchedulerBase::set_background(const std::vector<vtkm::Float32> &background)
{
  const size_t size = background.size();
  m_background.resize(size);

  for(size_t i = 0; i < size; ++i)
  {
    m_background[i] = static_cast<vtkm::Float64>(background[i]);
  }

}

void
SchedulerBase::clear_data_sets()
{
  m_domains.clear();
}

std::vector<Domain>
SchedulerBase::get_domains()
{
  return m_domains;
}

void
SchedulerBase::add_data_set(vtkmDataSet &dataset)
{
  ROVER_INFO("Adding domain "<<m_domains.size());
  Domain domain;
  domain.set_data_set(dataset);
  m_domains.push_back(domain);
}

vtkmDataSet
SchedulerBase::get_data_set(const int &domain)
{
  return m_domains.at(domain).get_data_set();
}

void
SchedulerBase::create_default_background(const int num_channels)
{
  m_background.resize(num_channels);
  for(int i = 0; i < num_channels; ++i)
  {
    m_background[i] = 1.f;
  }
}

void
SchedulerBase::set_domains(std::vector<Domain> &domains)
{
  m_domains = domains;
}

#ifdef ROVER_PARALLEL
void
SchedulerBase::set_comm_handle(MPI_Comm comm_handle)
{
  m_comm_handle = comm_handle;
}
#endif


} // namespace rover
