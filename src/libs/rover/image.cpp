//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "settings.hpp"
#include <image.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>

#include <viskores/cont/Field.h>

namespace rover
{

template<typename FloatType>
void
Image<FloatType>::normalize_handle(viskores::cont::ArrayHandle<FloatType> &handle,
                                   bool invert,
                                   float min_val,
                                   float max_val,
                                   bool log_scale)
{

  viskores::cont::Field as_field("name meaningless",
                             viskores::cont::Field::Association::Points,
                             handle);
  viskores::Range range;
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
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
  const int64 size = width * height;

#ifdef ROVER_OPENMP_ENABLED
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
Image<FloatType>::normalize_handle(viskores::cont::ArrayHandle<FloatType> &handle, bool invert)
{
  // TODO: Surely we can do better than "name meaningless"
  viskores::cont::Field as_field("name meaningless",
                             viskores::cont::Field::Association::Points,
                             handle);
  viskores::Range range;
  as_field.GetRange(&range);
  FloatType min_scalar = static_cast<FloatType>(range.Min);
  FloatType max_scalar = static_cast<FloatType>(range.Max);
  FloatType inv_delta;
  inv_delta = min_scalar == max_scalar ? 1.f : 1.f / (max_scalar - min_scalar);
  auto portal = handle.WritePortal();
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
  const int64 size = width * height;

#ifdef ROVER_OPENMP_ENABLED
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
{

}

template<typename FloatType>
Image<FloatType>::Image(PartialImage<FloatType> &partial)
{
  init_from_partial(partial);
}

template<typename FloatType>
void
Image<FloatType>::operator=(PartialImage<FloatType> partial)
{
  init_from_partial(partial);
}
//
// template specialization to handle the magic

template <typename T, typename O>
void cast_array_handle(viskores::cont::ArrayHandle<T> &cast_to,
                       viskores::cont::ArrayHandle<O> &cast_from)
{
  const viskores::Id size = cast_from.GetNumberOfValues();
  cast_to.Allocate(size);
  auto portal_to = cast_to.WritePortal();
  auto portal_from = cast_from.ReadPortal();
#ifdef ROVER_OPENMP_ENABLED
  #pragma omp parallel for
#endif
  for(viskores::Id i = 0; i < size; ++i)
  {
    portal_to.Set(i, static_cast<T>(portal_from.Get(i)));
  }
}
//
template<typename T, typename O> void init_from_image(Image<T> &left, Image<O> &right)
{
  const size_t num_energy_groups = right.m_intensity_values.size();
  for(size_t i = 0; i < num_energy_groups; ++i)
  {
    cast_array_handle(left.m_intensity_values[i], right.m_intensity_values[i]);
    cast_array_handle(left.m_optical_depth_values[i], right.m_optical_depth_values[i]);
  }

}
template<> void init_from_image<viskores::Float32, viskores::Float32>(Image<viskores::Float32> &left,
                                                              Image<viskores::Float32> &right)
{
  left.m_intensity_values = right.m_intensity_values;
  left.m_optical_depth_values = right.m_optical_depth_values;
}

template<> void init_from_image<viskores::Float64, viskores::Float64>(Image<viskores::Float64> &left,
                                                              Image<viskores::Float64> &right)
{
  left.m_intensity_values = right.m_intensity_values;
  left.m_optical_depth_values = right.m_optical_depth_values;
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
Image<FloatType>::get_num_energy_groups() const
{
  return static_cast<int>(m_intensity_values.size());
}

template<typename FloatType>
bool
Image<FloatType>::has_intensity(const int &channel_num) const
{
  return channel_num >= 0 && channel_num < m_intensity_values.size();
}

template<typename FloatType>
bool
Image<FloatType>::has_optical_depth(const int &channel_num) const
{
  return channel_num >= 0 && channel_num < m_optical_depth_values.size();
}

template<typename FloatType>
void
Image<FloatType>::init_from_partial(PartialImage<FloatType> &partial)
{
  m_intensity_values.clear();
  m_optical_depth_values.clear();

  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
  const int64 channel_size = width * height;
  const int num_energy_groups = partial.m_transmission.GetNumChannels();

  // Helper lambda to expand a channel and push its buffer to the output vector
  auto expand_and_push = [&](int channel_index,
                                       viskoresRayTracing::ChannelBuffer<FloatType>& channel_group,
                                       FloatType default_value,
                                       std::vector<HandleType>& output_vector)
  {
    viskoresRayTracing::ChannelBuffer<FloatType> channel = channel_group.GetChannel(channel_index);
    viskoresRayTracing::ChannelBuffer<FloatType> expanded = channel.ExpandBuffer(partial.m_pixel_ids, channel_size, default_value);
    output_vector.push_back(expanded.Buffer);
  };

  for (int i = 0; i < num_energy_groups; i++)
  {
    // Intensities
    expand_and_push(i,
                    partial.m_intensity,
                    partial.m_source_sig[i],
                    m_intensity_values);

    // Optical depths
    expand_and_push(i,
                    partial.m_optical_depth,
                    static_cast<FloatType>(0.0f),
                    m_optical_depth_values);
  }
}

template<typename FloatType>
viskores::cont::ArrayHandle<FloatType>
Image<FloatType>::get_intensity(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_intensity_values.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  return m_intensity_values[channel_num];
}

template<typename FloatType>
viskores::cont::ArrayHandle<FloatType>
Image<FloatType>::get_optical_depth(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_optical_depth_values.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  return m_optical_depth_values[channel_num];
}

template<typename FloatType>
viskores::cont::ArrayHandle<FloatType>
Image<FloatType>::flatten_intensity_values()
{
  const int num_energy_groups = get_num_energy_groups();

  HandleType res;
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
  const int64 size = width * height;
  res.Allocate(num_energy_groups * size);
  auto output = res.WritePortal();
  for(int c = 0; c < num_energy_groups; ++c)
  {
    auto channel = m_intensity_values[c].ReadPortal();

#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      // Deinterleave the output: all pixels for group 0, then all pixels for group 1, etc.
      output.Set(c * size + i, channel.Get(i));
    }
  }
  return res;
}

template<typename FloatType>
viskores::cont::ArrayHandle<FloatType>
Image<FloatType>::flatten_optical_depth_values()
{
  const int num_energy_groups = get_num_energy_groups();

  HandleType res;
  const int64 width = rover::settings["width"].to_int64();
  const int64 height = rover::settings["height"].to_int64();
  const int64 size = width * height;
  res.Allocate(num_energy_groups * size);
  auto output = res.WritePortal();
  for(int c = 0; c < num_energy_groups; ++c)
  {
    auto channel = m_optical_depth_values[c].ReadPortal();
#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
    for(int i = 0; i < size; ++i)
    {
      // Deinterleave the output: all pixels for group 0, then all pixels for group 1, etc.
      output.Set( c * size + i, channel.Get(i));
    }
  }
  return res;
}

template<typename FloatType>
void
Image<FloatType>::normalize_intensity(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_intensity_values.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  bool invert = false;
  normalize_handle(m_intensity_values[channel_num], invert);
}

template<typename FloatType>
void
Image<FloatType>::normalize_intensity(const int &channel_num,
                                      const float min_val,
                                      const float max_val,
                                      const bool log_scale)
{
  if(channel_num < 0 || channel_num >= m_intensity_values.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  bool invert = false;
  normalize_handle(m_intensity_values[channel_num], invert, min_val, max_val, log_scale);
}

template<typename FloatType>
void
Image<FloatType>::normalize_optical_depth(const int &channel_num)
{
  if(channel_num < 0 || channel_num >= m_optical_depth_values.size())
  {
    throw RoverException("Rover Image: invalid channel number");
  }
  bool invert = false;
  normalize_handle(m_optical_depth_values[channel_num], invert);
}
//
// Explicit instantiations
template class Image<viskores::Float32>;
template class Image<viskores::Float64>;

template void Image<viskores::Float32>::operator=<viskores::Float32>(Image<viskores::Float32> &other);
template void Image<viskores::Float32>::operator=<viskores::Float64>(Image<viskores::Float64> &other);
template void Image<viskores::Float64>::operator=<viskores::Float32>(Image<viskores::Float32> &other);
template void Image<viskores::Float64>::operator=<viskores::Float64>(Image<viskores::Float64> &other);

} // namespace rover
