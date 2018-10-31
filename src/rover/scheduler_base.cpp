//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-749865
// 
// All rights reserved.
// 
// This file is part of Rover. 
// 
// Please also read rover/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
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

#ifdef PARALLEL
void 
SchedulerBase::set_comm_handle(MPI_Comm comm_handle)
{
  m_comm_handle = comm_handle;
}
#endif


} // namespace rover
