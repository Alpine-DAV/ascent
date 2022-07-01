// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/color_map.hpp>

namespace dray
{

ColorMap::ColorMap ()
: m_color_table ("cool2warm"),
  m_samples (1024),
  m_log_scale (false),
  m_alpha_scale(1.f)
{
  m_color_table.sample (m_samples, m_colors, m_alpha_scale);
}

ColorMap::ColorMap (const std::string color_table)
: m_color_table (color_table),
  m_samples (1024),
  m_log_scale (false),
  m_alpha_scale(1.f)
{
  m_color_table.sample (m_samples, m_colors, m_alpha_scale);
}

void ColorMap::color_table (const ColorTable &color_table)
{
  m_color_table = color_table;
  m_color_table.sample (m_samples, m_colors, m_alpha_scale);
}

ColorTable ColorMap::color_table()
{
  return m_color_table;
}

void ColorMap::scalar_range (const Range &range)
{
  m_range = range;
}

Range ColorMap::scalar_range () const
{
  return m_range;
}

Array<Vec<float32, 4>> ColorMap::colors()
{
  return m_colors;
}

float32 ColorMap::alpha_scale() const
{
  return m_alpha_scale;
}

void ColorMap::alpha_scale(const float32 factor)
{
  m_alpha_scale = factor;
  m_color_table.sample (m_samples, m_colors, m_alpha_scale);
}

bool ColorMap::range_set()
{
  return !m_range.is_empty();
}

void ColorMap::log_scale (bool on)
{
  m_log_scale = on;
}

bool ColorMap::log_scale () const
{
  return m_log_scale;
}

void ColorMap::samples (int32 samples)
{
  assert (samples > 0);
  m_samples = samples;
  m_color_table.sample (m_samples, m_colors, m_alpha_scale);
}

void ColorMap::print()
{
  const Vec<float32,4> *colors = m_colors.get_host_ptr_const();
  const int size = m_colors.size();
  std::cout<<"********************************\n";
  for(int i = 0; i < size; ++i) std::cout<<i<<" "<<colors[i]<<"\n";
  std::cout<<"********************************\n";
}

} // namespace dray
