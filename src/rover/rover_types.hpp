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
#ifndef rover_types_h
#define rover_types_h

#include <vtkm_typedefs.hpp>

namespace rover {
// this could be ray tracing(surface) / volume rendering / energy
enum RenderMode
{
  volume, // standard volume rendering (3d)
  energy, // radiography with enegry bins (3d)
  surface //
};

enum ScatteringType
{
  scattering,    // ray can be scattered in random directions
  non_scattering // rays cannot be scattered
};
//
// Ray scope is only meaningful in parallel and is ignored otherwise
//
enum RayScope
{
  global_rays,  // rays can exist in one ore more domains at one time
  local_rays    // ran only exist in a single domain st any given time
};
//
// Volume rendering specific settigns
//
struct VolumeSettings
{
  int m_num_samples; // approximate number of samples per ray
  vtkmRange m_scalar_range;
  VolumeSettings()
    : m_num_samples(400)
  {}
};
//
// Energy specific settings
//
struct EnergySettings
{
  bool m_divide_abs_by_emmision;
  float m_unit_scalar;
  EnergySettings()
    : m_divide_abs_by_emmision(false),
      m_unit_scalar(1.0)
  {}
};

struct RenderSettings
{
  RenderMode     m_render_mode;
  ScatteringType m_scattering_type;
  RayScope       m_ray_scope;
  vtkmColorTable m_color_table;
  std::string    m_primary_field;
  std::string    m_secondary_field;
  VolumeSettings m_volume_settings;
  EnergySettings m_energy_settings;
  //
  // Default settings
  //
  RenderSettings()
    : m_color_table("cool2warm")
  {
    m_render_mode     = volume;
    m_scattering_type = non_scattering;
    m_ray_scope       = global_rays;
  }

  void print()
  {
    //std::cout<<"******* Settings *********\n";
    //if(m_render_mode == energy)
    //{
    //  std::cout<<"Render Mode: ENERGY\n";
    //}
    //else
    //{
    //  std::cout<<"Render Mode: VOLUME\n";
    //}

    //std::cout<<"Primary field: "<<m_primary_field<<"\n";
    //std::cout<<"Secondary field: "<<m_secondary_field<<"\n";
    //
    //std::cout<<"*************************\n";


  }
};

} // namespace rover
#endif
