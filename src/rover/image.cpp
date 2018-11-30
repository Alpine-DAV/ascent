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
#include <image.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>

#include <vtkm/cont/Field.h>

namespace rover
{

template<typename FloatType>
void
Image<FloatType>::normalize_handle(vtkm::cont::ArrayHandle<FloatType> &handle, bool invert)
{

  vtkm::cont::Field as_field("name meaningless",
                             vtkm::cont::Field::Association::POINTS,
                             handle);
  vtkm::Range range;
  as_field.GetRange(&range);
  FloatType min_scalar = static_cast<FloatType>(range.Min);
  FloatType max_scalar = static_cast<FloatType>(range.Max);
  FloatType inv_delta;
  inv_delta = min_scalar == max_scalar ? 1.f : 1.f / (max_scalar - min_scalar);
  auto portal = handle.GetPortalControl();
  const int size = m_width * m_height;
#ifdef ROVER_ENABLE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    FloatType val = portal.Get(i);
    val = (val - min_scalar) * inv_delta;
    if(invert) val = 1.f - val;
    portal.Set(i, val);
  }
}

template<typename FloatType>
Image<FloatType>::Image()
  : m_width(0),
    m_height(0)
{

}

template<typename FloatType>
Image<FloatType>::Image(PartialImage<FloatType> &partial)
{
  this->init_from_partial(partial);
}

template<typename FloatType>
void
Image<FloatType>::operator=(PartialImage<FloatType> partial)
{
  this->init_from_partial(partial);
}
//
// template specialization to handle the magic

template <typename T, typename O>
void cast_array_handle(vtkm::cont::ArrayHandle<T> &cast_to,
                        vtkm::cont::ArrayHandle<O> &cast_from)
{
  const vtkm::Id size = cast_from.GetNumberOfValues();
  cast_to.Allocate(size);
  auto portal_to = cast_to.GetPortalControl();
  auto portal_from = cast_to.GetPortalConstControl();
#ifdef ROVER_ENABLE_OPENMP
  #pragma omp parallel for
#endif
  for(vtkm::Id i = 0; i < size; ++i)
  {
    portal_to.Set(i, static_cast<T>(portal_from.Get(i)));
  }
}
//
template<typename T, typename O> void init_from_image(Image<T> &left, Image<O> &right)
{
  left.m_height = right.m_height;
  left.m_width = right.m_width;
  left.m_has_path_lengths = right.m_has_path_lengths;
  left.m_valid_intensities = right.m_valid_intensities;
  left.m_valid_optical_depths = right.m_valid_optical_depths;

  const size_t channels = right.m_intensities.size();
  for(size_t i = 0; i < channels; ++i)
  {
    cast_array_handle(left.m_intensities[i], right.m_intensities[i]);
    cast_array_handle(left.m_optical_depths[i], right.m_optical_depths[i]);
  }

  cast_array_handle(left.m_path_lengths,right.m_path_lengths);
}
template<> void init_from_image<vtkm::Float32, vtkm::Float32>(Image<vtkm::Float32> &left,
                                                              Image<vtkm::Float32> &right)
{
  left.m_height = right.m_height;;
  left.m_width = right.m_width;
  left.m_has_path_lengths = right.m_has_path_lengths;
  left.m_intensities = right.m_intensities;
  left.m_optical_depths = right.m_optical_depths;
  left.m_valid_intensities = right.m_valid_intensities;
  left.m_valid_optical_depths = right.m_valid_optical_depths;
  left.m_path_lengths = right.m_path_lengths;
}

template<> void init_from_image<vtkm::Float64, vtkm::Float64>(Image<vtkm::Float64> &left,
                                                              Image<vtkm::Float64> &right)
{
  left.m_height = right.m_height;;
  left.m_width = right.m_width;
  left.m_has_path_lengths = right.m_has_path_lengths;
  left.m_intensities = right.m_intensities;
  left.m_optical_depths = right.m_optical_depths;
  left.m_valid_intensities = right.m_valid_intensities;
  left.m_valid_optical_depths = right.m_valid_optical_depths;
  left.m_path_lengths = right.m_path_lengths;
}

template<typename FloatType>
template<typename O>
void
Image<FloatType>::operator=(Image<O> &other)
{
  init_from_image(*this,other);
}

template<typename FloatType>
int
Image<FloatType>::get_num_channels() const
{
  return static_cast<int>(m_intensities.size());
}

template<typename FloatType>
bool
Image<FloatType>::has_intensity(const int &channel_num) const
{
  if(channel_num < 0 || channel_num >= m_intensities.size())
  {
    return false;
  }

  if(!m_valid_intensities.at(channel_num))
  {
    return false;
  }

  return true;
}

template<typename FloatType>
bool
Image<FloatType>::has_optical_depth(const int &channel_num) const
{
  if(channel_num < 0 || channel_num >= m_optical_depths.size())
  {
    return false;
  }

  if(!m_valid_optical_depths.at(channel_num))
  {
    return false;
  }

  return true;
}

template<typename FloatType>
vtkm::cont::ArrayHandle<FloatType>
Image<FloatType>::get_path_lengths()
{
  if(!m_has_path_lengths)
  {
    throw RoverException("Rover Image: cannot get paths. They dont exist or have already been stolen.");
  }

  return m_path_lengths;
}

template<typename FloatType>
void
Image<FloatType>::normalize_paths()
{
  if(!m_has_path_lengths)
  {
    throw RoverException("Rover Image: cannot get paths. They dont exist or have already been stolen.");
  }
  bool invert = false;
  normalize_handle(m_path_lengths, false);
}

template<typename FloatType>
FloatType *
Image<FloatType>::steal_path_lengths()
{
  if(!m_has_path_lengths)
  {
    throw RoverException("Rover Image: cannot steal paths. They dont exist or have already been stolen.");
  }

  m_path_lengths.SyncControlArray();
  using StoreType = vtkm::cont::internal::Storage<FloatType, vtkm::cont::StorageTagBasic>;
  StoreType *storage = reinterpret_cast<StoreType*>(m_path_lengths.Internals->ControlArray);
  FloatType *ptr = reinterpret_cast<FloatType*>(storage->StealArray());
  m_has_path_lengths = false;
  return ptr;
}

template<typename FloatType>
bool
Image<FloatType>::has_path_lengths() const
{
  return m_has_path_lengths;
}

template<typename FloatType>
FloatType *
Image<FloatType>::steal_intensity(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_intensities.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }

  if(!m_valid_intensities.at(channel_num))
  {
    throw RoverException("Rover Image: cannot steal an instensity channel that has already been stolen");
  }
  m_intensities[channel_num].SyncControlArray();
  using StoreType = vtkm::cont::internal::Storage<FloatType, vtkm::cont::StorageTagBasic>;
  StoreType *storage = reinterpret_cast<StoreType*>(m_intensities[channel_num].Internals->ControlArray);
  FloatType *ptr = reinterpret_cast<FloatType*>(storage->StealArray());
  return ptr;
}

template<typename FloatType>
FloatType *
Image<FloatType>::steal_optical_depth(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_intensities.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }

  if(!m_valid_optical_depths.at(channel_num))
  {
    throw RoverException("Rover Image: cannot steal an optical depth channel that has already been stolen");
  }
  m_optical_depths[channel_num].SyncControlArray();
  using StoreType = vtkm::cont::internal::Storage<FloatType, vtkm::cont::StorageTagBasic>;
  StoreType *storage = reinterpret_cast<StoreType*>(m_optical_depths[channel_num].Internals->ControlArray);
  FloatType *ptr = reinterpret_cast<FloatType*>(storage->StealArray());
  return ptr;
}

template<typename FloatType>
void
Image<FloatType>::init_from_partial(PartialImage<FloatType> &partial)
{
  m_intensities.clear();
  m_optical_depths.clear();
  m_valid_intensities.clear();
  m_valid_optical_depths.clear();

  m_height = partial.m_height;
  m_width  = partial.m_width;
  assert(m_width > 0);
  assert(m_height > 0);
  m_has_path_lengths = partial.m_path_lengths.GetNumberOfValues() != 0;

  const int num_channels = partial.m_buffer.GetNumChannels();
  for(int i = 0; i < num_channels; ++i)
  {
    vtkmRayTracing::ChannelBuffer<FloatType> channel = partial.m_buffer.GetChannel( i );
    const FloatType default_value = partial.m_source_sig.size() != 0 ? partial.m_source_sig[i] : 0.0f;
    const int channel_size = m_height * m_width;
    vtkmRayTracing::ChannelBuffer<FloatType>  expand;
    expand = channel.ExpandBuffer(partial.m_pixel_ids,
                                  channel_size,
                                  default_value);

    m_optical_depths.push_back(expand.Buffer);
    m_valid_optical_depths.push_back(true);

  }

  for(int i = 0; i < num_channels; ++i)
  {
    vtkmRayTracing::ChannelBuffer<FloatType> channel = partial.m_intensities.GetChannel( i );
    const FloatType default_value = partial.m_source_sig.size() != 0 ? partial.m_source_sig[i] : 0.0f;
    const int channel_size = m_height * m_width;
    vtkmRayTracing::ChannelBuffer<FloatType>  expand;
    expand = channel.ExpandBuffer(partial.m_pixel_ids,
                                  channel_size,
                                  default_value);

    m_intensities.push_back(expand.Buffer);
    m_valid_intensities.push_back(true);

  }

  if(m_has_path_lengths)
  {
    const int size = m_width * m_height;
    m_path_lengths.Allocate(size);
    auto portal = m_path_lengths.GetPortalControl();
#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      portal.Set(i, 0.0f);
    }
    const int num_ids = static_cast<int>(partial.m_pixel_ids.GetNumberOfValues());
    auto id_portal = partial.m_pixel_ids.GetPortalControl();
    auto path_portal = partial.m_path_lengths.GetPortalControl();
#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < num_ids; ++i)
    {
      const int index = id_portal.Get(i);
      portal.Set(index, path_portal.Get(i));
    }
  }
}

template<typename FloatType>
vtkm::cont::ArrayHandle<FloatType>
Image<FloatType>::get_intensity(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_intensities.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  if(!m_valid_intensities.at(channel_num))
  {
    throw RoverException("Rover Image: cannot get an intensity that has already been stolen");
  }
  return m_intensities[channel_num];
}

template<typename FloatType>
vtkm::cont::ArrayHandle<FloatType>
Image<FloatType>::get_optical_depth(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_optical_depths.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  if(!m_valid_optical_depths.at(channel_num))
  {
    throw RoverException("Rover Image: cannot get an optical depth that has already been stolen");
  }
  return m_optical_depths[channel_num];
}

template<typename FloatType>
vtkm::cont::ArrayHandle<FloatType>
Image<FloatType>::flatten_intensities()
{
  const int num_channels = this->get_num_channels();
  for(int i = 0; i < num_channels; ++i)
  {
    if(!m_valid_intensities.at(i))
    {
      throw RoverException("Rover Image: cannot flatten intensities when channel has been stolen");
    }
  }
  HandleType res;
  const int size = m_width * m_height;
  res.Allocate(num_channels * size);
  auto output = res.GetPortalControl();
  for(int c = 0; c < num_channels; ++c)
  {
    auto channel = m_intensities[c].GetPortalControl();

#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      output.Set( i * num_channels + c, channel.Get(i));
    }
  }
  return res;
}

template<typename FloatType>
vtkm::cont::ArrayHandle<FloatType>
Image<FloatType>::flatten_optical_depths()
{
  const int num_channels = this->get_num_channels();
  for(int i = 0; i < num_channels; ++i)
  {
    if(!m_valid_optical_depths.at(i))
    {
      throw RoverException("Rover Image: cannot flatten optical depths when channel has been stolen");
    }
  }
  HandleType res;
  const int size = m_width * m_height;
  res.Allocate(num_channels * size);
  auto output = res.GetPortalControl();
  for(int c = 0; c < num_channels; ++c)
  {
    auto channel = m_optical_depths[c].GetPortalControl();
#ifdef ROVER_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      output.Set( i * num_channels + c, channel.Get(i));
    }
  }
  return res;
}

template<typename FloatType>
int
Image<FloatType>::get_size()
{
  return  m_width * m_height;
}

template<typename FloatType>
void
Image<FloatType>::normalize_intensity(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_intensities.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  if(!m_valid_intensities.at(channel_num))
  {
    throw RoverException("Rover Image: cannot normalize an intensity channel that has already been stolen");
  }
  bool invert = false;
  normalize_handle(m_intensities[channel_num], invert);
}

template<typename FloatType>
void
Image<FloatType>::normalize_optical_depth(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_optical_depths.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  if(!m_valid_optical_depths.at(channel_num))
  {
    throw RoverException("Rover Image: cannot normalize an optical depth channel that has already been stolen");
  }
  bool invert = false;
  normalize_handle(m_optical_depths[channel_num], invert);
}
//
// Explicit instantiations
template class Image<vtkm::Float32>;
template class Image<vtkm::Float64>;

template void Image<vtkm::Float32>::operator=<vtkm::Float32>(Image<vtkm::Float32> &other);
template void Image<vtkm::Float32>::operator=<vtkm::Float64>(Image<vtkm::Float64> &other);
template void Image<vtkm::Float64>::operator=<vtkm::Float32>(Image<vtkm::Float32> &other);
template void Image<vtkm::Float64>::operator=<vtkm::Float64>(Image<vtkm::Float64> &other);

} // namespace rover
