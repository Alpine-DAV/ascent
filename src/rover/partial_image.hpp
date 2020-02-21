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
#ifndef rover_partial_image_h
#define rover_partial_image_h

#include <vector>

#include <vtkm/cont/ArrayHandle.h>
#include <vtkh/compositing/AbsorptionPartial.hpp>
#include <vtkh/compositing/EmissionPartial.hpp>
#include <vtkh/compositing/VolumePartial.hpp>

namespace rover
{

template<typename FloatType>
struct PartialImage
{
  int                                      m_height;
  int                                      m_width;
  IdHandle                                 m_pixel_ids;
  vtkmRayTracing::ChannelBuffer<FloatType> m_buffer;          // holds either color or absorption
  vtkmRayTracing::ChannelBuffer<FloatType> m_intensities;     // holds the intensity emerging from each ray
  vtkm::cont::ArrayHandle<FloatType>       m_distances;
  std::vector<FloatType>                   m_source_sig;

  void allocate(const vtkm::Id &size, const vtkm::Id &channels)
  {
    m_pixel_ids.Allocate(size);
    m_distances.Allocate(size);
    m_buffer.SetNumChannels(channels);
    m_buffer.Resize(size);

    m_intensities.SetNumChannels(channels);
    m_intensities.Resize(size);
    m_source_sig.resize(channels);
  }

  PartialImage()
    : m_height(0),
      m_width(0)
  {

  }

  void extract_partials(std::vector<vtkh::VolumePartial<FloatType>> &partials)
  {
    auto id_portal = m_pixel_ids.GetPortalConstControl();
    auto buffer_portal = m_buffer.Buffer.GetPortalConstControl();
    auto depth_portal = m_distances.GetPortalConstControl();
    const int size = static_cast<int>(m_pixel_ids.GetNumberOfValues());
    partials.resize(size);

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      partials[i].m_pixel_id = static_cast<int>(id_portal.Get(i));
      partials[i].m_depth = static_cast<float>(depth_portal.Get(i));

      partials[i].m_pixel[0] = static_cast<float>(buffer_portal.Get(i*4+0));
      partials[i].m_pixel[1] = static_cast<float>(buffer_portal.Get(i*4+1));
      partials[i].m_pixel[2] = static_cast<float>(buffer_portal.Get(i*4+2));

      partials[i].m_alpha = static_cast<float>(buffer_portal.Get(i*4+3));
    }
  }

  void extract_partials(std::vector<vtkh::AbsorptionPartial<FloatType>> &partials)
  {
    const int num_bins = m_buffer.GetNumChannels();
    auto id_portal = m_pixel_ids.GetPortalConstControl();
    auto buffer_portal = m_buffer.Buffer.GetPortalConstControl();
    auto depth_portal = m_distances.GetPortalConstControl();
    const int size = static_cast<int>(m_pixel_ids.GetNumberOfValues());
    partials.resize(size);

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int index = 0; index < size; ++index)
    {
      partials[index].m_pixel_id = static_cast<int>(id_portal.Get(index));
      partials[index].m_depth = depth_portal.Get(index);
      partials[index].m_bins.resize(num_bins);

      const int starting_index = index * num_bins;
      for(int i = 0; i < num_bins; ++i)
      {
        partials[index].m_bins[i] = buffer_portal.Get(starting_index + i);
      }
    }
  }

  void extract_partials(std::vector<vtkh::EmissionPartial<FloatType>> &partials)
  {
    const int num_bins = m_buffer.GetNumChannels();
    auto id_portal = m_pixel_ids.GetPortalConstControl();
    auto buffer_portal = m_buffer.Buffer.GetPortalConstControl();
    auto intensity_portal = m_intensities.Buffer.GetPortalConstControl();

    auto depth_portal = m_distances.GetPortalConstControl();
    const int size = static_cast<int>(m_pixel_ids.GetNumberOfValues());

    partials.resize(size);

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int index = 0; index < size; ++index)
    {
      partials[index].m_pixel_id = static_cast<int>(id_portal.Get(index));
      partials[index].m_depth = depth_portal.Get(index);
      partials[index].m_bins.resize(num_bins);
      partials[index].m_emission_bins.resize(num_bins);

      const int starting_index = index * num_bins;
      for(int i = 0; i < num_bins; ++i)
      {
        partials[index].m_bins[i] = buffer_portal.Get(starting_index + i);
        partials[index].m_emission_bins[i] = intensity_portal.Get(starting_index + i);
      }
    }
  }

  void store(std::vector<vtkh::VolumePartial<FloatType>> &partials,
             const std::vector<double> &background,
             const int width,
             const int height)
  {
    m_width = width;
    m_height = height;
    const int size = static_cast<int>(partials.size());
    allocate(size,4);

    auto id_portal = m_pixel_ids.GetPortalControl();
    auto buffer_portal = m_buffer.Buffer.GetPortalControl();
    auto depth_portal = m_distances.GetPortalControl();
    auto intensity_portal = m_intensities.Buffer.GetPortalControl();

    vtkh::VolumePartial<FloatType> bg_color;
    bg_color.m_pixel[0] = static_cast<FloatType>(background[0]);
    bg_color.m_pixel[1] = static_cast<FloatType>(background[1]);
    bg_color.m_pixel[2] = static_cast<FloatType>(background[2]);
    bg_color.m_alpha    = static_cast<FloatType>(background[3]);

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      id_portal.Set(i, partials[i].m_pixel_id );
      depth_portal.Set(i, partials[i].m_depth );
      const int starting_index = i * 4;

      buffer_portal.Set(starting_index + 0, static_cast<FloatType>(partials[i].m_pixel[0]));
      buffer_portal.Set(starting_index + 1, static_cast<FloatType>(partials[i].m_pixel[1]));
      buffer_portal.Set(starting_index + 2, static_cast<FloatType>(partials[i].m_pixel[2]));
      buffer_portal.Set(starting_index + 3, static_cast<FloatType>(partials[i].m_alpha));

      partials[i].blend(bg_color);

      intensity_portal.Set(starting_index + 0, static_cast<FloatType>(partials[i].m_pixel[0]));
      intensity_portal.Set(starting_index + 1, static_cast<FloatType>(partials[i].m_pixel[1]));
      intensity_portal.Set(starting_index + 2, static_cast<FloatType>(partials[i].m_pixel[2]));
      intensity_portal.Set(starting_index + 3, static_cast<FloatType>(partials[i].m_alpha));
    }

    for(int i = 0; i < 4; ++i)
    {
      m_source_sig[i] = background[i];
    }

  }

  void store(std::vector<vtkh::AbsorptionPartial<FloatType>> &partials,
             const std::vector<double> &background,
             const int width,
             const int height)
  {
    m_width = width;
    m_height = height;
    const int size = static_cast<int>(partials.size());
    const int num_bins = static_cast<int>(partials.at(0).m_bins.size());
    allocate(size,num_bins);

    auto id_portal = m_pixel_ids.GetPortalControl();
    auto buffer_portal = m_buffer.Buffer.GetPortalControl();
    auto depth_portal = m_distances.GetPortalControl();
    auto intensity_portal = m_intensities.Buffer.GetPortalControl();

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      id_portal.Set(i, partials[i].m_pixel_id );
      depth_portal.Set(i, partials[i].m_depth );
      const int starting_index = i * num_bins;

      for(int ii = 0; ii < num_bins; ++ii)
      {
        buffer_portal.Set(starting_index + ii, partials[i].m_bins[ii]);
        intensity_portal.Set( starting_index + ii, partials[i].m_bins[ii] * background[ii]);
      }
    }

    for(int i = 0; i < num_bins; ++i)
    {
      m_source_sig[i] = background[i];
    }
  }

  void store(std::vector<vtkh::EmissionPartial<FloatType>> &partials,
             const std::vector<double> &background,
             const int width,
             const int height)
  {
    m_width = width;
    m_height = height;
    const int size = static_cast<int>(partials.size());
    const int num_bins = static_cast<int>(partials.at(0).m_bins.size());
    allocate(size,num_bins);

    auto id_portal = m_pixel_ids.GetPortalControl();
    auto buffer_portal = m_buffer.Buffer.GetPortalControl();
    auto depth_portal = m_distances.GetPortalControl();
    auto intensity_portal = m_intensities.Buffer.GetPortalControl();

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      id_portal.Set(i, partials[i].m_pixel_id );
      depth_portal.Set(i, partials[i].m_depth );
      const int starting_index = i * num_bins;

      for(int ii = 0; ii < num_bins; ++ii)
      {
        buffer_portal.Set(starting_index + ii, partials[i].m_bins[ii]);
        FloatType out_intensity;
        out_intensity = partials[i].m_emission_bins[ii] +  partials[i].m_bins[ii] * background[ii];
        intensity_portal.Set( starting_index + ii, out_intensity);
      }
    }

    for(int i = 0; i < num_bins; ++i)
    {
      m_source_sig[i] = background[i];
    }
  }

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
