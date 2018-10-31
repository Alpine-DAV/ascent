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
#ifndef rover_scheduler_base_h
#define rover_scheduler_base_h

#include <domain.hpp>
#include <image.hpp>
#include <engine.hpp>
#include <rover_types.hpp>
#include <ray_generators/ray_generator.hpp>
#include <vtkm_typedefs.hpp>

#ifdef PARALLEL
#include <mpi.h>
#endif
//
// Scheduler types:
//  static: all ranks gets all rays
//  normal compositing -
//    back to front (energy): absorbtion, absorbtion + emmission
//    front to back (volume): normal volume rendering
//  dynamic(scattering):
//    domain passing - 
//      front to back: volume rendering and ray tracing
//      back to front: both energy types.
//      
//
//
namespace rover {

class SchedulerBase 
{
public:
  SchedulerBase();
  virtual ~SchedulerBase();
  virtual void trace_rays() = 0;
  virtual void save_result(std::string file_name) = 0;
  void clear_data_sets();
  //
  // Setters
  //
  void set_render_settings(const RenderSettings render_settings);
  void add_data_set(vtkmDataSet &data_set);
  void set_domains(std::vector<Domain> &domains);
  void set_ray_generator(RayGenerator *ray_generator);
  void set_background(const std::vector<vtkm::Float32> &background);
  void set_background(const std::vector<vtkm::Float64> &background);
#ifdef PARALLEL
  void set_comm_handle(MPI_Comm comm_handle);
#endif
  //
  // Getters
  //
  std::vector<Domain> get_domains();
  RenderSettings get_render_settings() const;
  vtkmDataSet    get_data_set(const int &domain);
  virtual void get_result(Image<vtkm::Float32> &image) = 0;
  virtual void get_result(Image<vtkm::Float64> &image) = 0;
protected:
  std::vector<Domain>                       m_domains;
  RenderSettings                            m_render_settings;
  RayGenerator                             *m_ray_generator;
  std::vector<vtkm::Float64>                m_background;
  void create_default_background(const int num_channels);
#ifdef PARALLEL
  MPI_Comm                                  m_comm_handle;
#endif

};

}; // namespace rover
#endif 
