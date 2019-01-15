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
  bool           m_path_lengths;
  //
  // Default settings
  //
  RenderSettings()
    : m_color_table("cool2warm")
  {
    m_render_mode     = volume;
    m_scattering_type = non_scattering;
    m_ray_scope       = global_rays;
    m_path_lengths    = false;
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



template<typename FloatType>
struct PartialImage
{
  int                                      m_height;
  int                                      m_width;
  IdHandle                                 m_pixel_ids;
  vtkmRayTracing::ChannelBuffer<FloatType> m_buffer;          // holds either color or absorption
  vtkmRayTracing::ChannelBuffer<FloatType> m_intensities;     // holds the intensity emerging from each ray
  vtkm::cont::ArrayHandle<FloatType>       m_distances;
  vtkm::cont::ArrayHandle<FloatType>       m_path_lengths;
  std::vector<FloatType>                   m_source_sig;

  void add_source_sig()
  {
    auto buffer_portal = m_buffer.Buffer.GetPortalControl();
    auto int_portal = m_intensities.Buffer.GetPortalControl();
    const int size = m_pixel_ids.GetPortalControl().GetNumberOfValues();
    const int num_channels = m_buffer.GetNumChannels();

    bool has_emission = m_intensities.Buffer.GetNumberOfValues() != 0;
    if(!has_emission)
    {
      m_intensities.SetNumChannels(num_channels);
      m_intensities.Resize(size);
    }

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      for(int b = 0; b < num_channels; ++b)
      {
        const int offset = i * num_channels;
        FloatType emis = 0;
        if(has_emission)
        {
          emis = int_portal.Get(offset + b);
        }

        int_portal.Set(offset + b, emis + buffer_portal.Get(offset + b) * m_source_sig[b]);
      }
    }
  }

  void print_pixel(const int x, const int y)
  {
    const int size = m_pixel_ids.GetPortalControl().GetNumberOfValues();
    const int num_channels = m_buffer.GetNumChannels();
    bool has_emission = m_intensities.Buffer.GetNumberOfValues() != 0;
    int debug = m_width * ( m_height - y) + x;

    for(int i = 0; i < size; ++i)
    {
      if(m_pixel_ids.GetPortalControl().Get(i) == debug)
      {
        int offset = i * num_channels;
        for(int j = 0; j < num_channels ; ++j)
        {
          std::cout<<m_buffer.Buffer.GetPortalControl().Get(offset + j)<<" ";
          if(has_emission)
          {
            std::cout<<"("<<m_intensities.Buffer.GetPortalControl().Get(offset + j)<<") ";
          }
        }
        std::cout<<"\n";
      }
    }

  }// print

  void make_red_pixel(const int x, const int y)
    {
      const int size = m_pixel_ids.GetPortalControl().GetNumberOfValues();
      const int num_channels = m_buffer.GetNumChannels();
      int debug = m_width * ( m_height - y) + x;

      for(int i = 0; i < size; ++i)
      {
        if(m_pixel_ids.GetPortalControl().Get(i) == debug)
        {
          int offset = i * num_channels;
          m_buffer.Buffer.GetPortalControl().Set(offset , 1.f);
          for(int j = 1; j < num_channels -1; ++j)
          {
            m_buffer.Buffer.GetPortalControl().Set(offset + j, 0.f);
          }
          m_buffer.Buffer.GetPortalControl().Set(offset + num_channels-1,1.f);
        }
      }
    }
};

} // namespace rover
#endif
