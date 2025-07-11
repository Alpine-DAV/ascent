//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_partial_image_h
#define rover_partial_image_h

#include <rover_config.h>
#include <settings.hpp>
#include <vtkm_typedefs.hpp>

#include <vector>

#include <vtkm/cont/ArrayHandle.h>
#include <vtkh/compositing/AbsorptionPartial.hpp>
#include <vtkh/compositing/EmissionPartial.hpp>
// #include <vtkh/compositing/VolumePartial.hpp> // removing volume renderer

namespace rover
{

template<typename FloatType>
struct PartialImage
{
  // TODO: Improve naming to reflect what these things actually represent,
  // will require changes elsewhere (absorptionpartial and emissionpartial)
  IdHandle                                 m_pixel_ids;
  vtkmRayTracing::ChannelBuffer<FloatType> m_transmission;  // holds the fraction of incoming intensity that survives absorption
  vtkmRayTracing::ChannelBuffer<FloatType> m_intensity;     // holds the intensity emerging from each ray
  vtkmRayTracing::ChannelBuffer<FloatType> m_optical_depth;
  vtkm::cont::ArrayHandle<FloatType>       m_distances;
  std::vector<FloatType>                   m_source_sig;

  void allocate(const vtkm::Id &size, const vtkm::Id &channels)
  {
    m_pixel_ids.Allocate(size);
    m_distances.Allocate(size);
    m_transmission.SetNumChannels(channels);
    m_transmission.Resize(size);
    m_intensity.SetNumChannels(channels);
    m_intensity.Resize(size);
    m_optical_depth.SetNumChannels(channels);
    m_optical_depth.Resize(size);
    m_source_sig.resize(channels);
  }

  PartialImage()
  {

  }

#if 0 // removing volume renderer
  void extract_partials(std::vector<vtkh::VolumePartial<FloatType>> &partials)
  {
    auto id_portal = m_pixel_ids.ReadPortal();
    auto buffer_portal = m_buffer.Buffer.ReadPortal();
    auto depth_portal = m_distances.ReadPortal();
    const int size = static_cast<int>(m_pixel_ids.GetNumberOfValues());
    partials.resize(size);

#ifdef ROVER_OPENMP_ENABLED
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
#endif

  void extract_partials(std::vector<vtkh::AbsorptionPartial<FloatType>> &partials)
  {
    const int num_bins = m_transmission.GetNumChannels();
    auto id_portal = m_pixel_ids.ReadPortal();
    auto transmission_portal = m_transmission.Buffer.ReadPortal();
    auto depth_portal = m_distances.ReadPortal();
    const int size = static_cast<int>(m_pixel_ids.GetNumberOfValues());
    partials.resize(size);

#ifdef ROVER_OPENMP_ENABLED
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
        partials[index].m_bins[i] = transmission_portal.Get(starting_index + i);
      }
    }
  }

  void extract_partials(std::vector<vtkh::EmissionPartial<FloatType>> &partials)
  {
    const int num_bins = m_transmission.GetNumChannels();
    auto id_portal = m_pixel_ids.ReadPortal();
    auto transmission_portal = m_transmission.Buffer.ReadPortal();
    auto intensity_portal = m_intensity.Buffer.ReadPortal();
    auto optical_depth_portal = m_optical_depth.Buffer.ReadPortal();
    auto depth_portal = m_distances.ReadPortal();
    const int size = static_cast<int>(m_pixel_ids.GetNumberOfValues());
    partials.resize(size);

#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int index = 0; index < size; ++index)
    {
      partials[index].m_pixel_id = static_cast<int>(id_portal.Get(index));
      partials[index].m_depth = depth_portal.Get(index);
      partials[index].m_bins.resize(num_bins);
      partials[index].m_emission_bins.resize(num_bins);
      partials[index].m_optical_depth_bins.resize(num_bins);

      const int starting_index = index * num_bins;
      for(int i = 0; i < num_bins; ++i)
      {
        partials[index].m_bins[i] = transmission_portal.Get(starting_index + i);
        partials[index].m_emission_bins[i] = intensity_portal.Get(starting_index + i);
        partials[index].m_optical_depth_bins[i] = optical_depth_portal.Get(starting_index + i);
      }
    }
  }

#if 0 // removing volume renderer
  void store(std::vector<vtkh::VolumePartial<FloatType>> &partials,
             const std::vector<double> &background,
             const int width,
             const int height)
  {
    m_width = width;
    m_height = height;
    const int size = static_cast<int>(partials.size());
    allocate(size,4);

    auto id_portal = m_pixel_ids.WritePortal();
    auto buffer_portal = m_buffer.Buffer.WritePortal();
    auto depth_portal = m_distances.WritePortal();
    auto intensity_portal = m_intensities.Buffer.WritePortal();

    vtkh::VolumePartial<FloatType> bg_color;
    bg_color.m_pixel[0] = static_cast<FloatType>(background[0]);
    bg_color.m_pixel[1] = static_cast<FloatType>(background[1]);
    bg_color.m_pixel[2] = static_cast<FloatType>(background[2]);
    bg_color.m_alpha    = static_cast<FloatType>(background[3]);

#ifdef ROVER_OPENMP_ENABLED
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
#endif

  void store(std::vector<vtkh::AbsorptionPartial<FloatType>> &partials,
             const std::vector<double> &background)
  {
    const int size = static_cast<int>(partials.size());
    const int num_bins = static_cast<int>(partials.at(0).m_bins.size());
    allocate(size,num_bins);

    auto id_portal = m_pixel_ids.WritePortal();
    auto transmission_portal = m_transmission.Buffer.WritePortal();
    auto depth_portal = m_distances.WritePortal();
    auto intensity_portal = m_intensity.Buffer.WritePortal();

#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      id_portal.Set(i, partials[i].m_pixel_id);
      depth_portal.Set(i, partials[i].m_depth);
      const int starting_index = i * num_bins;

      for(int j = 0; j < num_bins; ++j)
      {
        transmission_portal.Set(starting_index + j, partials[i].m_bins[j]);
        intensity_portal.Set(starting_index + j, partials[i].m_bins[j] * background[j]);
      }
    }

    for(int i = 0; i < num_bins; ++i)
    {
      m_source_sig[i] = background[i];
    }
  }

  void store(std::vector<vtkh::EmissionPartial<FloatType>> &partials,
             const std::vector<double> &background)
  {
    const int size = static_cast<int>(partials.size());
    const int num_bins = static_cast<int>(partials.at(0).m_bins.size());
    allocate(size,num_bins);

    auto id_portal = m_pixel_ids.WritePortal();
    auto transmission_portal = m_transmission.Buffer.WritePortal();
    auto depth_portal = m_distances.WritePortal();
    auto intensity_portal = m_intensity.Buffer.WritePortal();
    auto optical_depth_portal = m_optical_depth.Buffer.WritePortal();

#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      id_portal.Set(i, partials[i].m_pixel_id);
      depth_portal.Set(i, partials[i].m_depth);
      const int starting_index = i * num_bins;

      for(int j = 0; j < num_bins; ++j)
      {
        const int starting_index_j = starting_index + j;
        // TODO: Add comment explaining what's going on here
        transmission_portal.Set(starting_index_j, partials[i].m_bins[j]);
        intensity_portal.Set(starting_index_j, partials[i].m_emission_bins[j] + partials[i].m_bins[j] * background[j]);
        optical_depth_portal.Set(starting_index_j, partials[i].m_optical_depth_bins[j]);
      }
    }

    for(int i = 0; i < num_bins; ++i)
    {
      m_source_sig[i] = background[i];
    }
  }

  void add_source_sig()
  {
    auto buffer_portal = m_transmission.Buffer.WritePortal();
    auto int_portal = m_intensity.Buffer.WritePortal();
    const int size = m_pixel_ids.GetNumberOfValues();
    const int num_channels = m_transmission.GetNumChannels();

    bool has_emission = m_intensity.Buffer.GetNumberOfValues() != 0;
    if(!has_emission)
    {
      m_intensity.SetNumChannels(num_channels);
      m_intensity.Resize(size);
    }

#ifdef ROVER_OPENMP_ENABLED
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
};

} // namespace rover
#endif
