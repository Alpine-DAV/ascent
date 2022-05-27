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
#include "PartialCompositor.hpp"
#include <algorithm>
#include <assert.h>
#include <limits>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#include "vtkh_diy_partial_redistribute.hpp"
#include "vtkh_diy_partial_collect.hpp"
#endif

namespace vtkh {
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
  //
  // Perform the compositing and output the result in the output
  //
#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for
#endif
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
  //
  // Perform the compositing and output the result in the output
  // This code computes the optical depth (total absorption)
  // along each rays path.
  //
#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for
#endif
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
#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for
#endif
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
PartialCompositor<PartialType>::PartialCompositor()
{

}

//--------------------------------------------------------------------------------------------

template<typename PartialType>
PartialCompositor<PartialType>::~PartialCompositor()
{

}

//--------------------------------------------------------------------------------------------

template<typename PartialType>
void
PartialCompositor<PartialType>::merge(const std::vector<std::vector<PartialType>> &in_partials,
                               std::vector<PartialType> &partials,
                               int &global_min_pixel,
                               int &global_max_pixel)
{

  int total_partial_comps = 0;
  const int num_partial_images = static_cast<int>(in_partials.size());
  int *offsets = new int[num_partial_images];
  int *pixel_mins =  new int[num_partial_images];
  int *pixel_maxs =  new int[num_partial_images];

  for(int i = 0; i < num_partial_images; ++i)
  {
    offsets[i] = total_partial_comps;
    total_partial_comps += in_partials[i].size();
  }

  partials.resize(total_partial_comps);

#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < num_partial_images; ++i)
  {
    //
    //  Extract the partial composites into a contiguous array
    //
    std::copy(in_partials[i].begin(), in_partials[i].end(), partials.begin() + offsets[i]);
  }// for each partial image

  //
  // Calculate the range of pixel ids
  //
  int max_pixel = std::numeric_limits<int>::min();
#ifdef VTKH_USE_OPENMP
    #pragma omp parallel for reduction(max:max_pixel)
#endif
  for(int i = 0; i < total_partial_comps; ++i)
  {
    int val = partials[i].m_pixel_id;
    if(val > max_pixel)
    {
      max_pixel = val;
    }
  }

   int min_pixel = std::numeric_limits<int>::max();
#ifdef VTKH_USE_OPENMP
    #pragma omp parallel for reduction(min:min_pixel)
#endif
  for(int i = 0; i < total_partial_comps; ++i)
  {
    int val = partials[i].m_pixel_id;
    if(val < min_pixel)
    {
      min_pixel = val;
    }
  }

  //
  // determine the global pixel mins and maxs
  //
  global_min_pixel = min_pixel;
  global_max_pixel = max_pixel;

#ifdef VTKH_PARALLEL
  MPI_Comm comm_handle = MPI_Comm_f2c(m_mpi_comm_id);
  int rank_min = global_min_pixel;
  int rank_max = global_max_pixel;
  int mpi_min;
  int mpi_max;
  MPI_Allreduce(&rank_min, &mpi_min, 1, MPI_INT, MPI_MIN, comm_handle);
  MPI_Allreduce(&rank_max, &mpi_max, 1, MPI_INT, MPI_MAX, comm_handle);
  global_min_pixel = mpi_min;
  global_max_pixel = mpi_max;
#endif

  delete[] offsets;
  delete[] pixel_mins;
  delete[] pixel_maxs;

}

//--------------------------------------------------------------------------------------------
template<typename PartialType>
void
PartialCompositor<PartialType>::composite_partials(std::vector<PartialType> &partials,
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

#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for
#endif
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

#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for shared(work_flags) reduction(+:total_segments)
#endif
  for(int i = 0; i < total_partial_comps; ++i)
  {
    total_segments += work_flags[i];
  }

  int total_unique_pixels = 0;
#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for shared(unique_flags) reduction(+:total_unique_pixels)
#endif
  for(int i = 0; i < total_partial_comps; ++i)
  {
    total_unique_pixels += unique_flags[i];
  }

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

  output_partials.resize(total_output_pixels);

  //
  // Gather the unique pixels into the output
  //
#ifdef VTKH_USE_OPENMP
  #pragma omp parallel for
#endif
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
void
PartialCompositor<PartialType>::composite(std::vector<std::vector<PartialType>> &partial_images,
                                   std::vector<PartialType> &output_partials)
{
  int global_partial_images = partial_images.size();
#ifdef VTKH_PARALLEL
  MPI_Comm comm_handle = MPI_Comm_f2c(m_mpi_comm_id);
  int local_partials = global_partial_images;
  MPI_Allreduce(&local_partials, &global_partial_images, 1, MPI_INT, MPI_SUM, comm_handle);
#endif

#ifdef VTKH_PARALLEL
  // we could have no data, but it could exist elsewhere
#endif

  std::vector<PartialType> partials;
  int global_min_pixel;
  int global_max_pixel;

  merge(partial_images, partials, global_min_pixel, global_max_pixel);

  if(global_min_pixel > global_max_pixel)
  {
    // just bail
    return;
  }


#ifdef VTKH_PARALLEL
  //
  // Exchange partials with other ranks
  //
  redistribute(partials,
               comm_handle,
               global_min_pixel,
               global_max_pixel);
  MPI_Barrier(comm_handle);
#endif

  const int  total_partial_comps = partials.size();

  //
  // TODO: check to see if we have less than one
  //
  //assert(total_partial_comps > 1);

  composite_partials(partials, output_partials);

#ifdef VTKH_PARALLEL
  //
  // Collect all of the distibuted pixels
  //
  collect(output_partials, comm_handle);
  MPI_Barrier(comm_handle);
#endif
}

template<typename PartialType>
void
PartialCompositor<PartialType>::set_background(std::vector<vtkm::Float32> &background_values)
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
PartialCompositor<PartialType>::set_background(std::vector<vtkm::Float64> &background_values)
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
PartialCompositor<PartialType>::set_comm_handle(int mpi_comm_id)
{
  m_mpi_comm_id  = mpi_comm_id;
}

//Explicit function instantiations
template class VTKH_API PartialCompositor<VolumePartial<vtkm::Float32>>;
template class VTKH_API PartialCompositor<VolumePartial<vtkm::Float64>>;

template class VTKH_API PartialCompositor<AbsorptionPartial<vtkm::Float32>>;
template class VTKH_API PartialCompositor<AbsorptionPartial<vtkm::Float64>>;

template class VTKH_API PartialCompositor<EmissionPartial<vtkm::Float32>>;
template class VTKH_API PartialCompositor<EmissionPartial<vtkm::Float64>>;


} // namespace vtkh
