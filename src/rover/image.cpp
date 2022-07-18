//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <image.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>

#include <vtkm/cont/Field.h>

namespace rover
{

template<typename FloatType>
void
Image<FloatType>::normalize_handle(vtkm::cont::ArrayHandle<FloatType> &handle,
                                   bool invert,
                                   float min_val,
                                   float max_val,
                                   bool log_scale)
{

  vtkm::cont::Field as_field("name meaningless",
                             vtkm::cont::Field::Association::Points,
                             handle);
  vtkm::Range range;
  as_field.GetRange(&range);
  FloatType min_scalar = static_cast<FloatType>(min_val);
  FloatType max_scalar = static_cast<FloatType>(max_val);
  if(min_scalar > max_scalar)
  {
    throw RoverException("Rover Image: min_value > max_value");
  }
  if(log_scale)
  {
    if(min_scalar <= 0.f)
    {
      throw RoverException("Rover Image: log scale range contains values <= 0");
    }
    min_scalar = log(min_scalar);
    max_scalar = log(max_scalar);
  }

  FloatType inv_delta;
  inv_delta = min_scalar == max_scalar ? 1.f : 1.f / (max_scalar - min_scalar);
  auto portal = handle.WritePortal();
  const int size = m_width * m_height;
#ifdef ROVER_ENABLE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < size; ++i)
  {
    FloatType val = portal.Get(i);
    if(log_scale)
    {
      val = log(val);
    }
    val = fmin(max_scalar, fmax(val, min_scalar));
    val = (val - min_scalar) * inv_delta;
    if(invert) val = 1.f - val;
    portal.Set(i, val);
  }
}

template<typename FloatType>
void
Image<FloatType>::normalize_handle(vtkm::cont::ArrayHandle<FloatType> &handle, bool invert)
{

  vtkm::cont::Field as_field("name meaningless",
                             vtkm::cont::Field::Association::Points,
                             handle);
  vtkm::Range range;
  as_field.GetRange(&range);
  FloatType min_scalar = static_cast<FloatType>(range.Min);
  FloatType max_scalar = static_cast<FloatType>(range.Max);
  FloatType inv_delta;
  inv_delta = min_scalar == max_scalar ? 1.f : 1.f / (max_scalar - min_scalar);
  auto portal = handle.WritePortal();
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
  auto portal_to = cast_to.WritePortal();
  auto portal_from = cast_to.ReadPortal();
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
  left.m_valid_intensities = right.m_valid_intensities;
  left.m_valid_optical_depths = right.m_valid_optical_depths;

  const size_t channels = right.m_intensities.size();
  for(size_t i = 0; i < channels; ++i)
  {
    cast_array_handle(left.m_intensities[i], right.m_intensities[i]);
    cast_array_handle(left.m_optical_depths[i], right.m_optical_depths[i]);
  }

}
template<> void init_from_image<vtkm::Float32, vtkm::Float32>(Image<vtkm::Float32> &left,
                                                              Image<vtkm::Float32> &right)
{
  left.m_height = right.m_height;;
  left.m_width = right.m_width;
  left.m_intensities = right.m_intensities;
  left.m_optical_depths = right.m_optical_depths;
  left.m_valid_intensities = right.m_valid_intensities;
  left.m_valid_optical_depths = right.m_valid_optical_depths;
}

template<> void init_from_image<vtkm::Float64, vtkm::Float64>(Image<vtkm::Float64> &left,
                                                              Image<vtkm::Float64> &right)
{
  left.m_height = right.m_height;;
  left.m_width = right.m_width;
  left.m_intensities = right.m_intensities;
  left.m_optical_depths = right.m_optical_depths;
  left.m_valid_intensities = right.m_valid_intensities;
  left.m_valid_optical_depths = right.m_valid_optical_depths;
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
void
Image<FloatType>::init_from_partial(PartialImage<FloatType> &partial)
{
  m_intensities.clear();
  m_optical_depths.clear();
  m_valid_intensities.clear();
  m_valid_optical_depths.clear();

  m_height = partial.m_height;
  m_width  = partial.m_width;

  assert(m_width >= 0);
  assert(m_height >= 0);

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
  auto output = res.WritePortal();
  for(int c = 0; c < num_channels; ++c)
  {
    auto channel = m_intensities[c].ReadPortal();

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
  auto output = res.WritePortal();
  for(int c = 0; c < num_channels; ++c)
  {
    auto channel = m_optical_depths[c].ReadPortal();
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
Image<FloatType>::normalize_intensity(const int &channel_num,
                                      const float min_val,
                                      const float max_val,
                                      const bool log_scale)
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
  normalize_handle(m_intensities[channel_num], invert, min_val, max_val, log_scale);
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
