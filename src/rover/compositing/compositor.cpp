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
#include <utils/rover_logging.hpp>
#include <compositing/compositor.hpp>
#include <algorithm>
#include <assert.h>
#include <limits>

#ifdef PARALLEL
#include <compositing/redistribute.hpp>
#include <compositing/collect.hpp>
#endif

namespace rover {
namespace detail
{
template<template <typename> class PartialType, typename FloatType>
void BlendPartials(const int &total_segments, 
                   const int &total_partial_comps,
                   std::vector<int> &pixel_work_ids,
                   std::vector<PartialType<FloatType>> &partials,
                   std::vector<PartialType<FloatType>> &output_partials,
                   const int output_offset)
{
  ROVER_INFO("Blending partials volume or absoption");
  //
  // Perform the compositing and output the result in the output 
  //
  #pragma omp parallel for
  for(int i = 0; i < total_segments; ++i)
  {
    int current_index = pixel_work_ids[i];
    PartialType<FloatType> result = partials[current_index];
    ++current_index;
    PartialType<FloatType> next = partials[current_index];
    // TODO: we could just count the amount of work and make this a for loop(vectorize??)
    while(result.m_pixel_id == next.m_pixel_id)
    {
      result.blend(next);
      if(current_index + 1 >= total_partial_comps) 
      {
        // we could break early for volumes,
        // but blending past 1.0 alpha is no op.
        break;
      }
      ++current_index;
      next = partials[current_index];
    }
    output_partials[output_offset + i] = result;
  }

  //placeholder 
  //PartialType<FloatType>::composite_background(output_partials, background_values);

}
template<typename T>
void
BlendEmission(const int &total_segments, 
              const int &total_partial_comps,
              std::vector<int> &pixel_work_ids,
              std::vector<EmissionPartial<T>> &partials,
              std::vector<EmissionPartial<T>> &output_partials,
              const int output_offset)
{
  ROVER_INFO("Blending partials with emission");
  //
  // Perform the compositing and output the result in the output 
  // This code computes the optical depth (total absorption)
  // along each rays path.
  //
  #pragma omp parallel for
  for(int i = 0; i < total_segments; ++i)
  {
    int current_index = pixel_work_ids[i];
    EmissionPartial<T> result = partials[current_index];
    ++current_index;
    EmissionPartial<T> next = partials[current_index];
    // TODO: we could just count the amount of work and make this a for loop(vectorize??)
    while(result.m_pixel_id == next.m_pixel_id)
    {
      result.blend_absorption(next);
      if(current_index == total_partial_comps - 1) 
      {
        break;
      }
      ++current_index;
      next = partials[current_index];
    }
    output_partials[output_offset + i] = result;
  }

  //placeholder 
  //EmissionPartial::composite_background(output_partials);
  // TODO: now blend source signature with output 
  
  //
  //  Emission bins contain the amout of energy that leaves each
  //  ray segment. To compute the amount of energy that reaches 
  //  the detector, we must multiply the segments emissed energy
  //  by the optical depth of the remaining path to the detector.
  //  To calculate the optical depth of the remaining path, we
  //  do perform a reverse scan of absorption for each pixel id 
  //
  #pragma omp parallel for
  for(int i = 0; i < total_segments; ++i)
  {
    const int segment_start = pixel_work_ids[i];
    int current_index = segment_start;
    //
    //  move forward to the end of the segment
    //
    while(partials[current_index].m_pixel_id == partials[current_index + 1].m_pixel_id)
    {
      ++current_index;
      if(current_index == total_partial_comps - 1)
      {
        break;
      }
    }
    //
    // set the intensity emerging out of the last segment
    //
    output_partials[output_offset + i].m_emission_bins 
      = partials[current_index].m_emission_bins;
   
    //
    // now move backwards accumulating absorption for each segment
    // and then blending the intensity emerging from the previous 
    // segment.
    //
    current_index--;
    while(current_index != segment_start - 1)
    {
      partials[current_index].blend_absorption(partials[current_index + 1]);  
      // mult this segments emission by the absorption in front
      partials[current_index].blend_emission(partials[current_index + 1]);  
      // add remaining emissed engery to the output 
      output_partials[output_offset + i].add_emission(partials[current_index]);

      --current_index;
    }
  }


}
template<>
void BlendPartials<EmissionPartial, float>(const int &total_segments, 
                                           const int &total_partial_comps,
                                           std::vector<int> &pixel_work_ids,
                                           std::vector<EmissionPartial<float>> &partials,
                                           std::vector<EmissionPartial<float>> &output_partials,
                                           const int output_offset)
{

  BlendEmission(total_segments, 
                total_partial_comps,
                pixel_work_ids,
                partials,
                output_partials,
                output_offset);
}

template<>
void BlendPartials<EmissionPartial, double>(const int &total_segments, 
                                            const int &total_partial_comps,
                                            std::vector<int> &pixel_work_ids,
                                            std::vector<EmissionPartial<double>> &partials,
                                            std::vector<EmissionPartial<double>> &output_partials,
                                            const int output_offset)
{

  BlendEmission(total_segments, 
                total_partial_comps,
                pixel_work_ids,
                partials,
                output_partials,
                output_offset);
}

} // namespace detail

//--------------------------------------------------------------------------------------------
template<typename PartialType>
Compositor<PartialType>::Compositor()
{

}

//--------------------------------------------------------------------------------------------

template<typename PartialType>
Compositor<PartialType>::~Compositor()
{

}

//--------------------------------------------------------------------------------------------

template<typename PartialType>
void 
Compositor<PartialType>::extract(std::vector<PartialImage<typename PartialType::ValueType>> &partial_images, 
                          std::vector<PartialType> &partials,
                          int &global_min_pixel,
                          int &global_max_pixel)
{
  vtkmTimer tot_timer;  
  vtkmTimer timer;  
  double time = 0;
  ROVER_DATA_OPEN("compositing_extract");

  int total_partial_comps = 0;
  const int num_partial_images = static_cast<int>(partial_images.size());
  int *offsets = new int[num_partial_images];
  int *pixel_mins =  new int[num_partial_images];
  int *pixel_maxs =  new int[num_partial_images];

  for(int i = 0; i < num_partial_images; ++i)
  {
    //assert(partial_images[i].m_buffer.GetNumChannels() == 4);
    offsets[i] = total_partial_comps;
    total_partial_comps += partial_images[i].m_buffer.GetSize();
    ROVER_INFO("Domain : image  "<<i<<" with "<<partial_images[i].m_buffer.GetSize());
  }

  ROVER_INFO("Total number of partial composites "<<total_partial_comps);

  partials.resize(total_partial_comps);

  timer.Reset();

  for(int i = 0; i < num_partial_images; ++i)
  {
    //
    //  Extract the partial composites into a contiguous array
    //

    vtkmTimer timer1;  
    const int image_size = partial_images[i].m_buffer.GetSize();
    #pragma omp parallel for
    for(int j = 0; j < image_size; ++j)
    {
      int index = offsets[i] + j;
      partials[index].load_from_partial(partial_images[i], j);
    }
    ROVER_DATA_ADD("load from partials",timer1.GetElapsedTime()); 
    timer1.Reset();
    //
    // Calculate the range of pixel ids each domain has
    //
    auto id_portal = partial_images[i].m_pixel_ids.GetPortalConstControl();
    int max_pixel = std::numeric_limits<int>::min();
    #pragma omp parallel for reduction(max:max_pixel)
    for(int j = 0; j < image_size; ++j)
    {
      int val = static_cast<int>(id_portal.Get(j));
      if(val > max_pixel)
      {
        max_pixel = val;
      }
    }
    ROVER_DATA_ADD("max_pixel",timer1.GetElapsedTime()); 
    timer1.Reset();

    int min_pixel = std::numeric_limits<int>::max();
    #pragma omp parallel for reduction(min:min_pixel)
    for(int j = 0; j < image_size; ++j)
    {
      
      int val = static_cast<int>(id_portal.Get(j));
      if(val < min_pixel)
      {
        min_pixel = val;
      }

      assert(min_pixel > -1);
      pixel_mins[i] = min_pixel;
      pixel_maxs[i] = max_pixel;
    }

    ROVER_DATA_ADD("min_pixel",timer1.GetElapsedTime()); 
    timer1.Reset();
  }// for each partial image
  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("merge_partials",time); 
  timer.Reset();
  // 
  // determine the global pixel mins and maxs
  //
  global_min_pixel = std::numeric_limits<int>::max();
  global_max_pixel = std::numeric_limits<int>::min();
  for(int i = 0; i < num_partial_images; ++i)
  {
    global_min_pixel = std::min(global_min_pixel, pixel_mins[i]);   
    global_max_pixel = std::max(global_max_pixel, pixel_maxs[i]);   
  }

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("local_pixels",time); 
  timer.Reset();
#ifdef PARALLEL
  int rank_min = global_min_pixel;
  int rank_max = global_max_pixel;
  int mpi_min;
  int mpi_max;
  MPI_Allreduce(&rank_min, &mpi_min, 1, MPI_INT, MPI_MIN, m_comm_handle);
  MPI_Allreduce(&rank_max, &mpi_max, 1, MPI_INT, MPI_MAX, m_comm_handle);
  global_min_pixel = mpi_min;
  global_max_pixel = mpi_max;
#endif

  time = timer.GetElapsedTime();
  ROVER_DATA_ADD("global_pixels",time); 

  timer.Reset();

  delete[] offsets;
  delete[] pixel_mins;
  delete[] pixel_maxs;

  time = tot_timer.GetElapsedTime();
  ROVER_DATA_CLOSE(time);
}

//--------------------------------------------------------------------------------------------
template<typename PartialType>
void 
Compositor<PartialType>::composite_partials(std::vector<PartialType> &partials, 
                                            std::vector<PartialType> &output_partials)
{
  const int total_partial_comps = partials.size();
  if(total_partial_comps == 0)
  {
    output_partials = partials;
    return;
  }
  //
  // Sort the composites
  //
  std::sort(partials.begin(), partials.end());  
  ROVER_INFO("Sorted partials");
  // 
  // Find the number of unique pixel_ids with work
  //
  std::vector<unsigned char> work_flags;
  std::vector<unsigned char> unique_flags;
  work_flags.resize(total_partial_comps);
  unique_flags.resize(total_partial_comps);
  //
  // just check the first and last entries manualy to reduce the
  // loop complexity
  //
  if(partials[0].m_pixel_id == partials[1].m_pixel_id)
  {
    work_flags[0] = 1;
    unique_flags[0] = 0;
  }
  else
  {
    work_flags[0] = 0;
    unique_flags[0] = 1;
  }
  if(partials[total_partial_comps-1].m_pixel_id != partials[total_partial_comps-2].m_pixel_id)
  {
    unique_flags[total_partial_comps-1] = 1;
  }
  else
  {
    unique_flags[total_partial_comps-1] = 0;
  }
  const int n_minus_one =  total_partial_comps - 1;

  #pragma omp parallel for
  for(int i = 1; i < n_minus_one; ++i)
  {
    unsigned char work_flag = 0;
    unsigned char unique_flag = 0;
    bool is_begining = false;
    if(partials[i].m_pixel_id != partials[i-1].m_pixel_id)
    {
      is_begining = true;
    }

    bool has_compositing_work = false;
    if(partials[i].m_pixel_id == partials[i+1].m_pixel_id)
    {
      has_compositing_work = true;
    }
    if(is_begining && has_compositing_work)
    {
      work_flag  = 1;
    }
    if(is_begining && !has_compositing_work)
    {
      unique_flag = 1;
    }

    work_flags[i]  = work_flag;
    unique_flags[i] = unique_flag;
  }
  // count the number of of unique pixels
  int total_segments = 0;
  #pragma omp parallel for shared(work_flags) reduction(+:total_segments)
  for(int i = 0; i < total_partial_comps; ++i)
  {
    total_segments += work_flags[i];
  }

  int total_unique_pixels = 0;
  #pragma omp parallel for shared(unique_flags) reduction(+:total_unique_pixels)
  for(int i = 0; i < total_partial_comps; ++i)
  {
    total_unique_pixels += unique_flags[i];
  }

  ROVER_INFO("Total pixels that need compositing "<<total_segments<<" total partials "<<total_partial_comps);
  ROVER_INFO("Total unique pixels "<<total_unique_pixels);

  if(total_segments ==  0)
  {
    //nothing to do
  }
 
  //
  // find the pixel indexes that have compositing work
  //
  std::vector<int> pixel_work_ids;
  pixel_work_ids.resize(total_segments);
  int current_index = 0;
  for(int i = 0;  i < total_partial_comps; ++i)
  {
    if(work_flags[i] == 1)
    {
      pixel_work_ids[current_index] = i;
      ++current_index;
    }
  }

  //
  // find the pixel indexes that have NO compositing work
  //
  std::vector<int> unique_ids;
  unique_ids.resize(total_unique_pixels);
  current_index = 0;
  for(int i = 0;  i < total_partial_comps; ++i)
  {
    if(unique_flags[i] == 1)
    {
      unique_ids[current_index] = i;
      ++current_index;
    }
  }


  const int total_output_pixels = total_unique_pixels + total_segments;
  ROVER_INFO("Total output size "<<total_output_pixels);

  output_partials.resize(total_output_pixels);
  
  // 
  // Gather the unique pixels into the output
  //
  #pragma omp parallel for 
  for(int i = 0; i < total_unique_pixels; ++i)
  {
    PartialType result = partials[unique_ids[i]];
    output_partials[i] = result;
  }
  
  //
  // perform compositing if there are more than
  // one segment per ray
  //
  detail::BlendPartials(total_segments, 
                        total_partial_comps,
                        pixel_work_ids,
                        partials,
                        output_partials,
                        total_unique_pixels);

}

//--------------------------------------------------------------------------------------------

template<typename PartialType>
PartialImage<typename PartialType::ValueType> 
Compositor<PartialType>::composite(std::vector<PartialImage<typename PartialType::ValueType>> &partial_images)
{
  ROVER_INFO("Compsositor start");
  int global_partial_images = partial_images.size();
#ifdef PARALLEL
  int local_partials = global_partial_images;
  MPI_Allreduce(&local_partials, &global_partial_images, 1, MPI_INT, MPI_SUM, m_comm_handle);
#endif
  // there should always be at least one ray cast, 
  // so this should be a safe check
  bool has_path_lengths = false;
  if(partial_images.size() > 0)
  {
    has_path_lengths = partial_images[0].m_path_lengths.GetNumberOfValues() != 0;
  }

#ifdef PARALLEL
  // we could have no data, but it could exist elsewhere 
#endif

  ROVER_DATA_OPEN("compositing");
  vtkmTimer tot_timer; 
  vtkmTimer timer; 
  double time = 0;

  std::vector<PartialType> partials;
  int global_min_pixel;
  int global_max_pixel;

  ROVER_INFO("Extracing");
  extract(partial_images, partials, global_min_pixel, global_max_pixel);
  time = timer.GetElapsedTime(); 
  ROVER_DATA_ADD("extract", time);
  timer.Reset();

#ifdef PARALLEL
  //
  // Exchange partials with other ranks
  //
  redistribute(partials, 
               m_comm_handle,
               global_min_pixel,
               global_max_pixel);
  ROVER_INFO("Redistributed");
  MPI_Barrier(m_comm_handle);
#endif

  time = timer.GetElapsedTime(); 
  ROVER_DATA_ADD("redistribute", time);
  timer.Reset();

  const int  total_partial_comps = partials.size();

  ROVER_INFO("Extracted partial structs "<<total_partial_comps);

  //
  // TODO: check to see if we have less than one
  //
  //assert(total_partial_comps > 1);
  
  std::vector<PartialType> output_partials;
  composite_partials(partials, output_partials);
   
  time = timer.GetElapsedTime(); 
  ROVER_DATA_ADD("do_composite", time);
  timer.Reset();
#ifdef PARALLEL
  //
  // Collect all of the distibuted pixels
  //
  collect(output_partials, m_comm_handle);
  MPI_Barrier(m_comm_handle);
#endif
  
  time = timer.GetElapsedTime(); 
  ROVER_DATA_ADD("collect", time);
  timer.Reset();
  //
  // pack the output back into a channel buffer
  //
  const int num_channels = partial_images[0].m_buffer.GetNumChannels();
  PartialImage<typename PartialType::ValueType> output;
  output.m_width = partial_images[0].m_width;
  output.m_height= partial_images[0].m_height;
#ifdef PARALLEL
  int rank;
  MPI_Comm_rank(m_comm_handle, &rank);
  if(rank != 0)
  {
    ROVER_INFO("Bailing out of compositing");
    return output;
  }
#endif
  const int out_size = output_partials.size();
  //TODO make parital image init/allocate method
  ROVER_INFO("Allocating out buffers size "<<out_size);
  output.m_pixel_ids.Allocate(out_size);
  output.m_distances.Allocate(out_size);
  output.m_buffer.SetNumChannels(num_channels);
  output.m_buffer.Resize(out_size);

  output.m_intensities.SetNumChannels(num_channels);
  output.m_intensities.Resize(out_size);

  if(has_path_lengths)
  {
    ROVER_INFO("Allocating path lengths "<<out_size);
    output.m_path_lengths.Allocate(out_size);
  }

  #pragma omp parallel for
  for(int i = 0; i < out_size; ++i)
  {
    output_partials[i].store_into_partial(output, i, m_background_values);
  }

  ROVER_INFO("Compositing results in "<<out_size);

  time = timer.GetElapsedTime(); 
  ROVER_DATA_ADD("pack_partial", time);

  time = tot_timer.GetElapsedTime(); 
  ROVER_DATA_CLOSE(time);
  output.m_source_sig = m_background_values;
  output.m_width = partial_images[0].m_width;
  output.m_height = partial_images[0].m_height;
  return output;
}

template<typename PartialType>
void 
Compositor<PartialType>::set_background(std::vector<vtkm::Float32> &background_values)
{
  const size_t size = background_values.size();
  m_background_values.resize(size);
  for(size_t i = 0; i < size; ++i)
  {
    m_background_values[i] = background_values[i];
  }
}

template<typename PartialType>
void 
Compositor<PartialType>::set_background(std::vector<vtkm::Float64> &background_values)
{
  const size_t size = background_values.size();
  m_background_values.resize(size);
  for(size_t i = 0; i < size; ++i)
  {
    m_background_values[i] = background_values[i];
  }
}

#ifdef PARALLEL
template<typename PartialType>
void 
Compositor<PartialType>::set_comm_handle(MPI_Comm comm_handle)
{
  m_comm_handle = comm_handle;
}
#endif

//Explicit function instantiations
template class Compositor<VolumePartial<vtkm::Float32>>;
template class Compositor<VolumePartial<vtkm::Float64>>;

template class Compositor<AbsorptionPartial<vtkm::Float32>>;
template class Compositor<AbsorptionPartial<vtkm::Float64>>;

template class Compositor<EmissionPartial<vtkm::Float32>>;
template class Compositor<EmissionPartial<vtkm::Float64>>;


} // namespace rover
