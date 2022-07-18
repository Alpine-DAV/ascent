// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2016 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//  Copyright 2016 UT-Battelle, LLC.
//  Copyright 2016 Los Alamos National Security.
//
//  Under the terms of Contract DE-NA0003525 with NTESS,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

#include <dray/color_table.hpp>
#include <dray/additional_color_tables.hpp>

#include <string>
#include <vector>

namespace dray
{
namespace detail
{


struct ColorControlPoint
{
  float32 m_position;
  Vec<float32, 4> m_rgba;
  ColorControlPoint (float32 position, const Vec<float32, 4> &rgba)
  : m_position (position), m_rgba (rgba)
  {
  }
};

struct AlphaControlPoint
{
  float32 m_position;
  float32 m_alpha_value;
  AlphaControlPoint (float32 position, const float32 &alpha)
  : m_position (position), m_alpha_value (alpha)
  {
  }
};

struct ColorTableInternals
{
  std::string m_name;
  bool m_smooth;
  std::vector<ColorControlPoint> m_rgb_points;
  std::vector<AlphaControlPoint> m_alpha_points;
};

} // namespace detail

ColorTable::ColorTable () : m_internals (new detail::ColorTableInternals)
{
  this->m_internals->m_name = "";
  this->m_internals->m_smooth = false;
}

int ColorTable::number_of_points () const
{
  return m_internals->m_rgb_points.size ();
}

int ColorTable::number_of_alpha_points () const
{
  return m_internals->m_alpha_points.size ();
}

void ColorTable::update_alpha (const int32 &index, const float32 &position, const float32 &alpha)
{
  assert (index >= 0);
  assert (index < number_of_alpha_points ());
  assert (position >= 0.f);
  assert (position <= 1.f);
  assert (alpha >= 0.f);
  assert (alpha <= 1.f);
  detail::AlphaControlPoint &p = m_internals->m_alpha_points[index];

  p.m_position = position;
  p.m_alpha_value = alpha;
}

Vec<float32, 2> ColorTable::get_alpha (int index) const
{
  assert (index >= 0);
  assert (index < number_of_alpha_points ());
  // std::cout<<"index "<<index<<" number_of_alphas "<<number_of_alpha_points()<<"\n";
  const detail::AlphaControlPoint &p = m_internals->m_alpha_points[index];
  Vec<float32, 2> res;
  res[0] = p.m_position;
  res[1] = p.m_alpha_value;
  return res;
}

void ColorTable::remove_alpha (int index)
{
  assert (index >= 0);
  assert (index < number_of_alpha_points ());
  m_internals->m_alpha_points.erase (m_internals->m_alpha_points.begin () + index);
}

void ColorTable::print ()
{
  std::cout << "Color table: " << get_name () << "\n";
  const std::size_t num_points = this->m_internals->m_rgb_points.size ();
  const std::size_t num_alpha_points = this->m_internals->m_alpha_points.size ();

  for (int i = 0; i < num_points; ++i)
  {
    const Vec<float32, 4> &color = this->m_internals->m_rgb_points[i].m_rgba;
    const float32 pos = this->m_internals->m_rgb_points[i].m_position;
    std::cout << pos << " " << color << "\n";
  }
  std::cout << "Alphas :\n";
  for (int i = 0; i < num_alpha_points; ++i)
  {
    const float32 alpha = this->m_internals->m_alpha_points[i].m_alpha_value;
    const float32 pos = this->m_internals->m_alpha_points[i].m_position;
    std::cout << pos << " " << alpha << "\n";
  }
}

const std::string &ColorTable::get_name () const
{
  return this->m_internals->m_name;
}

bool ColorTable::get_smooth () const
{
  return this->m_internals->m_smooth;
}

void ColorTable::set_smooth (bool smooth)
{
  this->m_internals->m_smooth = smooth;
}

void ColorTable::sample (int num_samples,
                         Array<Vec<float32, 4>> &colors,
                         float32 alpha_scale) const
{
  colors.resize (num_samples);
  Vec<float32, 4> *color_ptr = colors.get_host_ptr ();
  for (int32 i = 0; i < num_samples; i++)
  {
    Vec<float32, 4> c =
    map_rgb (static_cast<float32> (i) / static_cast<float32> (num_samples - 1));
    c[3] = map_alpha (static_cast<float32> (i) / static_cast<float32> (num_samples - 1));
    c[3] *= alpha_scale;
    color_ptr[i] = c;
  }
}

Vec<float32, 4> ColorTable::map_rgb (float32 scalar) const
{
  std::size_t num_points = this->m_internals->m_rgb_points.size ();
  if (num_points == 0)
  {
    return make_vec4f (0.5f, 0.5f, 0.5f, 1.f);
  }
  if ((num_points == 1) || (scalar <= this->m_internals->m_rgb_points[0].m_position))
  {
    return this->m_internals->m_rgb_points[0].m_rgba;
  }
  if (scalar >= this->m_internals->m_rgb_points[num_points - 1].m_position)
  {
    return this->m_internals->m_rgb_points[num_points - 1].m_rgba;
  }

  std::size_t second_color_index;
  for (second_color_index = 1; second_color_index < num_points - 1; second_color_index++)
  {
    if (scalar < this->m_internals->m_rgb_points[second_color_index].m_position)
    {
      break;
    }
  }

  std::size_t first_color_index = second_color_index - 1;
  float32 seg = this->m_internals->m_rgb_points[second_color_index].m_position -
                this->m_internals->m_rgb_points[first_color_index].m_position;
  float32 alpha;
  if (seg == 0.f)
  {
    alpha = .5f;
  }
  else
  {
    alpha = (scalar - this->m_internals->m_rgb_points[first_color_index].m_position) / seg;
  }

  const Vec<float32, 4> &first_color =
  this->m_internals->m_rgb_points[first_color_index].m_rgba;
  const Vec<float32, 4> &second_color =
  this->m_internals->m_rgb_points[second_color_index].m_rgba;
  if (this->m_internals->m_smooth)
  {
    return (first_color * (1.0f - alpha) + second_color * alpha);
  }
  else
  {
    if (alpha < .5)
    {
      return first_color;
    }
    else
    {
      return second_color;
    }
  }
}

float32 ColorTable::map_alpha (float32 scalar) const
{
  std::size_t num_points = this->m_internals->m_alpha_points.size ();
  // If no alpha control points were set, just return full opacity
  if (num_points == 0)
  {
    return 1.f;
  }
  if ((num_points == 1) || (scalar <= this->m_internals->m_alpha_points[0].m_position))
  {
    return this->m_internals->m_alpha_points[0].m_alpha_value;
  }
  if (scalar >= this->m_internals->m_alpha_points[num_points - 1].m_position)
  {
    return this->m_internals->m_alpha_points[num_points - 1].m_alpha_value;
  }

  std::size_t second_color_index;
  for (second_color_index = 1; second_color_index < num_points - 1; second_color_index++)
  {
    if (scalar < this->m_internals->m_alpha_points[second_color_index].m_position)
    {
      break;
    }
  }

  std::size_t first_color_index = second_color_index - 1;
  float32 seg = this->m_internals->m_alpha_points[second_color_index].m_position -
                this->m_internals->m_alpha_points[first_color_index].m_position;
  float32 alpha;
  if (seg == 0.f)
  {
    alpha = .5;
  }
  else
  {
    alpha = (scalar - this->m_internals->m_alpha_points[first_color_index].m_position) / seg;
  }

  float32 first_alpha = this->m_internals->m_alpha_points[first_color_index].m_alpha_value;
  float32 second_alpha = this->m_internals->m_alpha_points[second_color_index].m_alpha_value;
  if (this->m_internals->m_smooth)
  {
    return (first_alpha * (1.f - alpha) + second_alpha * alpha);
  }
  else
  {
    if (alpha < .5)
    {
      return first_alpha;
    }
    else
    {
      return second_alpha;
    }
  }
}

void ColorTable::clear()
{
  this->m_internals->m_name = "";
  this->m_internals->m_alpha_points.clear();
  this->m_internals->m_rgb_points.clear();
}

void ColorTable::clear_colors()
{
  this->m_internals->m_name = "";
  this->m_internals->m_rgb_points.clear();
}

void ColorTable::clear_alphas()
{
  this->m_internals->m_name = "";
  this->m_internals->m_alpha_points.clear();
}

ColorTable ColorTable::correct_opacity (const float32 &factor) const
{
  ColorTable corrected;
  corrected.set_smooth (this->m_internals->m_smooth);
  size_t rgb_size = this->m_internals->m_rgb_points.size ();
  for (size_t i = 0; i < rgb_size; ++i)
  {
    detail::ColorControlPoint point = this->m_internals->m_rgb_points.at (i);
    corrected.add_point (point.m_position, point.m_rgba);
  }

  size_t alpha_size = this->m_internals->m_alpha_points.size ();
  for (size_t i = 0; i < alpha_size; ++i)
  {
    detail::AlphaControlPoint point = this->m_internals->m_alpha_points.at (i);
    float32 alpha = 1.f - pow ((1.f - point.m_alpha_value), factor);
    corrected.add_alpha (point.m_position, alpha);
  }

  return corrected;
}

void ColorTable::reverse ()
{
  std::shared_ptr<detail::ColorTableInternals> old_internals = this->m_internals;

  this->m_internals = std::make_shared<detail::ColorTableInternals>();

  int vector_size = old_internals->m_rgb_points.size ();
  for (int i = vector_size - 1; i >= 0; --i)
  {
    add_point (1.0f - old_internals->m_rgb_points[i].m_position,
               old_internals->m_rgb_points[i].m_rgba);
  }

  vector_size = old_internals->m_alpha_points.size ();
  for (int i = vector_size - 1; i >= 0; --i)
  {
    add_alpha (1.0f - old_internals->m_alpha_points[i].m_position,
               old_internals->m_alpha_points[i].m_alpha_value);
  }

  this->m_internals->m_smooth = old_internals->m_smooth;
  this->m_internals->m_name = old_internals->m_name;
}

void ColorTable::add_point (float32 position, const Vec<float32, 4> &color)
{
  this->m_internals->m_rgb_points.push_back (detail::ColorControlPoint (position, color));
}

void ColorTable::add_point (float32 position, const Vec<float32, 3> &color)
{
  Vec<float32, 4> c4;
  c4[0] = color[0];
  c4[1] = color[1];
  c4[2] = color[2];
  c4[3] = 1.0;
  this->m_internals->m_rgb_points.push_back (detail::ColorControlPoint (position, c4));
}

// void ColorTable::add_point(float32 position,
//                           const Vec<float32,4> &color,
//                           float32 alpha)
//{
//  this->m_internals->m_rgb_points.push_back(detail::ColorControlPoint(position, color));
//  this->m_internals->m_alpha_points.push_back(detail::AlphaControlPoint(position, alpha));
//}

void ColorTable::add_alpha (float32 position, float32 alpha)
{
  this->m_internals->m_alpha_points.push_back (detail::AlphaControlPoint (position, alpha));
}

ColorTable::ColorTable (const std::string &name_)
: m_internals (new detail::ColorTableInternals)
{
  std::string name = name_;
  if (name == "" || name == "default")
  {
    name = "cool2warm";
  }

  this->m_internals->m_smooth = true;
  if (name == "grey" || name == "gray")
  {
    add_point (0.0f, make_vec3f (0.f, 0.f, 0.f));
    add_point (1.0f, make_vec3f (1.f, 1.f, 1.f));
  }
  else if (name == "blue")
  {
    add_point (0.00f, make_vec3f (0.f, 0.f, 0.f));
    add_point (0.33f, make_vec3f (0.f, 0.f, .5f));
    add_point (0.66f, make_vec3f (0.f, .5f, 1.f));
    add_point (1.00f, make_vec3f (1.f, 1.f, 1.f));
  }
  else if (name == "orange")
  {
    add_point (0.00f, make_vec3f (0.f, 0.f, 0.f));
    add_point (0.33f, make_vec3f (.5f, 0.f, 0.f));
    add_point (0.66f, make_vec3f (1.f, .5f, 0.f));
    add_point (1.00f, make_vec3f (1.f, 1.f, 1.f));
  }
  else if (name == "cool2warm")
  {
    add_point (0.0f, make_vec3f (0.3347f, 0.2830f, 0.7564f));
    add_point (0.0039f, make_vec3f (0.3389f, 0.2901f, 0.7627f));
    add_point (0.0078f, make_vec3f (0.3432f, 0.2972f, 0.7688f));
    add_point (0.0117f, make_vec3f (0.3474f, 0.3043f, 0.7749f));
    add_point (0.0156f, make_vec3f (0.3516f, 0.3113f, 0.7809f));
    add_point (0.0196f, make_vec3f (0.3558f, 0.3183f, 0.7869f));
    add_point (0.0235f, make_vec3f (0.3600f, 0.3253f, 0.7928f));
    add_point (0.0274f, make_vec3f (0.3642f, 0.3323f, 0.7986f));
    add_point (0.0313f, make_vec3f (0.3684f, 0.3392f, 0.8044f));
    add_point (0.0352f, make_vec3f (0.3727f, 0.3462f, 0.8101f));
    add_point (0.0392f, make_vec3f (0.3769f, 0.3531f, 0.8157f));
    add_point (0.0431f, make_vec3f (0.3811f, 0.3600f, 0.8213f));
    add_point (0.0470f, make_vec3f (0.3853f, 0.3669f, 0.8268f));
    add_point (0.0509f, make_vec3f (0.3896f, 0.3738f, 0.8322f));
    add_point (0.0549f, make_vec3f (0.3938f, 0.3806f, 0.8375f));
    add_point (0.0588f, make_vec3f (0.3980f, 0.3874f, 0.8428f));
    add_point (0.0627f, make_vec3f (0.4023f, 0.3942f, 0.8480f));
    add_point (0.0666f, make_vec3f (0.4065f, 0.4010f, 0.8531f));
    add_point (0.0705f, make_vec3f (0.4108f, 0.4078f, 0.8582f));
    add_point (0.0745f, make_vec3f (0.4151f, 0.4145f, 0.8632f));
    add_point (0.0784f, make_vec3f (0.4193f, 0.4212f, 0.8680f));
    add_point (0.0823f, make_vec3f (0.4236f, 0.4279f, 0.8729f));
    add_point (0.0862f, make_vec3f (0.4279f, 0.4346f, 0.8776f));
    add_point (0.0901f, make_vec3f (0.4321f, 0.4412f, 0.8823f));
    add_point (0.0941f, make_vec3f (0.4364f, 0.4479f, 0.8868f));
    add_point (0.0980f, make_vec3f (0.4407f, 0.4544f, 0.8913f));
    add_point (0.1019f, make_vec3f (0.4450f, 0.4610f, 0.8957f));
    add_point (0.1058f, make_vec3f (0.4493f, 0.4675f, 0.9001f));
    add_point (0.1098f, make_vec3f (0.4536f, 0.4741f, 0.9043f));
    add_point (0.1137f, make_vec3f (0.4579f, 0.4805f, 0.9085f));
    add_point (0.1176f, make_vec3f (0.4622f, 0.4870f, 0.9126f));
    add_point (0.1215f, make_vec3f (0.4666f, 0.4934f, 0.9166f));
    add_point (0.1254f, make_vec3f (0.4709f, 0.4998f, 0.9205f));
    add_point (0.1294f, make_vec3f (0.4752f, 0.5061f, 0.9243f));
    add_point (0.1333f, make_vec3f (0.4796f, 0.5125f, 0.9280f));
    add_point (0.1372f, make_vec3f (0.4839f, 0.5188f, 0.9317f));
    add_point (0.1411f, make_vec3f (0.4883f, 0.5250f, 0.9352f));
    add_point (0.1450f, make_vec3f (0.4926f, 0.5312f, 0.9387f));
    add_point (0.1490f, make_vec3f (0.4970f, 0.5374f, 0.9421f));
    add_point (0.1529f, make_vec3f (0.5013f, 0.5436f, 0.9454f));
    add_point (0.1568f, make_vec3f (0.5057f, 0.5497f, 0.9486f));
    add_point (0.1607f, make_vec3f (0.5101f, 0.5558f, 0.9517f));
    add_point (0.1647f, make_vec3f (0.5145f, 0.5618f, 0.9547f));
    add_point (0.1686f, make_vec3f (0.5188f, 0.5678f, 0.9577f));
    add_point (0.1725f, make_vec3f (0.5232f, 0.5738f, 0.9605f));
    add_point (0.1764f, make_vec3f (0.5276f, 0.5797f, 0.9633f));
    add_point (0.1803f, make_vec3f (0.5320f, 0.5856f, 0.9659f));
    add_point (0.1843f, make_vec3f (0.5364f, 0.5915f, 0.9685f));
    add_point (0.1882f, make_vec3f (0.5408f, 0.5973f, 0.9710f));
    add_point (0.1921f, make_vec3f (0.5452f, 0.6030f, 0.9733f));
    add_point (0.1960f, make_vec3f (0.5497f, 0.6087f, 0.9756f));
    add_point (0.2f, make_vec3f (0.5541f, 0.6144f, 0.9778f));
    add_point (0.2039f, make_vec3f (0.5585f, 0.6200f, 0.9799f));
    add_point (0.2078f, make_vec3f (0.5629f, 0.6256f, 0.9819f));
    add_point (0.2117f, make_vec3f (0.5673f, 0.6311f, 0.9838f));
    add_point (0.2156f, make_vec3f (0.5718f, 0.6366f, 0.9856f));
    add_point (0.2196f, make_vec3f (0.5762f, 0.6420f, 0.9873f));
    add_point (0.2235f, make_vec3f (0.5806f, 0.6474f, 0.9890f));
    add_point (0.2274f, make_vec3f (0.5850f, 0.6528f, 0.9905f));
    add_point (0.2313f, make_vec3f (0.5895f, 0.6580f, 0.9919f));
    add_point (0.2352f, make_vec3f (0.5939f, 0.6633f, 0.9932f));
    add_point (0.2392f, make_vec3f (0.5983f, 0.6685f, 0.9945f));
    add_point (0.2431f, make_vec3f (0.6028f, 0.6736f, 0.9956f));
    add_point (0.2470f, make_vec3f (0.6072f, 0.6787f, 0.9967f));
    add_point (0.2509f, make_vec3f (0.6116f, 0.6837f, 0.9976f));
    add_point (0.2549f, make_vec3f (0.6160f, 0.6887f, 0.9985f));
    add_point (0.2588f, make_vec3f (0.6205f, 0.6936f, 0.9992f));
    add_point (0.2627f, make_vec3f (0.6249f, 0.6984f, 0.9999f));
    add_point (0.2666f, make_vec3f (0.6293f, 0.7032f, 1.0004f));
    add_point (0.2705f, make_vec3f (0.6337f, 0.7080f, 1.0009f));
    add_point (0.2745f, make_vec3f (0.6381f, 0.7127f, 1.0012f));
    add_point (0.2784f, make_vec3f (0.6425f, 0.7173f, 1.0015f));
    add_point (0.2823f, make_vec3f (0.6469f, 0.7219f, 1.0017f));
    add_point (0.2862f, make_vec3f (0.6513f, 0.7264f, 1.0017f));
    add_point (0.2901f, make_vec3f (0.6557f, 0.7308f, 1.0017f));
    add_point (0.2941f, make_vec3f (0.6601f, 0.7352f, 1.0016f));
    add_point (0.2980f, make_vec3f (0.6645f, 0.7395f, 1.0014f));
    add_point (0.3019f, make_vec3f (0.6688f, 0.7438f, 1.0010f));
    add_point (0.3058f, make_vec3f (0.6732f, 0.7480f, 1.0006f));
    add_point (0.3098f, make_vec3f (0.6775f, 0.7521f, 1.0001f));
    add_point (0.3137f, make_vec3f (0.6819f, 0.7562f, 0.9995f));
    add_point (0.3176f, make_vec3f (0.6862f, 0.7602f, 0.9988f));
    add_point (0.3215f, make_vec3f (0.6905f, 0.7641f, 0.9980f));
    add_point (0.3254f, make_vec3f (0.6948f, 0.7680f, 0.9971f));
    add_point (0.3294f, make_vec3f (0.6991f, 0.7718f, 0.9961f));
    add_point (0.3333f, make_vec3f (0.7034f, 0.7755f, 0.9950f));
    add_point (0.3372f, make_vec3f (0.7077f, 0.7792f, 0.9939f));
    add_point (0.3411f, make_vec3f (0.7119f, 0.7828f, 0.9926f));
    add_point (0.3450f, make_vec3f (0.7162f, 0.7864f, 0.9912f));
    add_point (0.3490f, make_vec3f (0.7204f, 0.7898f, 0.9897f));
    add_point (0.3529f, make_vec3f (0.7246f, 0.7932f, 0.9882f));
    add_point (0.3568f, make_vec3f (0.7288f, 0.7965f, 0.9865f));
    add_point (0.3607f, make_vec3f (0.7330f, 0.7998f, 0.9848f));
    add_point (0.3647f, make_vec3f (0.7372f, 0.8030f, 0.9829f));
    add_point (0.3686f, make_vec3f (0.7413f, 0.8061f, 0.9810f));
    add_point (0.3725f, make_vec3f (0.7455f, 0.8091f, 0.9789f));
    add_point (0.3764f, make_vec3f (0.7496f, 0.8121f, 0.9768f));
    add_point (0.3803f, make_vec3f (0.7537f, 0.8150f, 0.9746f));
    add_point (0.3843f, make_vec3f (0.7577f, 0.8178f, 0.9723f));
    add_point (0.3882f, make_vec3f (0.7618f, 0.8205f, 0.9699f));
    add_point (0.3921f, make_vec3f (0.7658f, 0.8232f, 0.9674f));
    add_point (0.3960f, make_vec3f (0.7698f, 0.8258f, 0.9648f));
    add_point (0.4f, make_vec3f (0.7738f, 0.8283f, 0.9622f));
    add_point (0.4039f, make_vec3f (0.7777f, 0.8307f, 0.9594f));
    add_point (0.4078f, make_vec3f (0.7817f, 0.8331f, 0.9566f));
    add_point (0.4117f, make_vec3f (0.7856f, 0.8353f, 0.9536f));
    add_point (0.4156f, make_vec3f (0.7895f, 0.8375f, 0.9506f));
    add_point (0.4196f, make_vec3f (0.7933f, 0.8397f, 0.9475f));
    add_point (0.4235f, make_vec3f (0.7971f, 0.8417f, 0.9443f));
    add_point (0.4274f, make_vec3f (0.8009f, 0.8437f, 0.9410f));
    add_point (0.4313f, make_vec3f (0.8047f, 0.8456f, 0.9376f));
    add_point (0.4352f, make_vec3f (0.8085f, 0.8474f, 0.9342f));
    add_point (0.4392f, make_vec3f (0.8122f, 0.8491f, 0.9306f));
    add_point (0.4431f, make_vec3f (0.8159f, 0.8507f, 0.9270f));
    add_point (0.4470f, make_vec3f (0.8195f, 0.8523f, 0.9233f));
    add_point (0.4509f, make_vec3f (0.8231f, 0.8538f, 0.9195f));
    add_point (0.4549f, make_vec3f (0.8267f, 0.8552f, 0.9156f));
    add_point (0.4588f, make_vec3f (0.8303f, 0.8565f, 0.9117f));
    add_point (0.4627f, make_vec3f (0.8338f, 0.8577f, 0.9076f));
    add_point (0.4666f, make_vec3f (0.8373f, 0.8589f, 0.9035f));
    add_point (0.4705f, make_vec3f (0.8407f, 0.8600f, 0.8993f));
    add_point (0.4745f, make_vec3f (0.8441f, 0.8610f, 0.8950f));
    add_point (0.4784f, make_vec3f (0.8475f, 0.8619f, 0.8906f));
    add_point (0.4823f, make_vec3f (0.8508f, 0.8627f, 0.8862f));
    add_point (0.4862f, make_vec3f (0.8541f, 0.8634f, 0.8817f));
    add_point (0.4901f, make_vec3f (0.8574f, 0.8641f, 0.8771f));
    add_point (0.4941f, make_vec3f (0.8606f, 0.8647f, 0.8724f));
    add_point (0.4980f, make_vec3f (0.8638f, 0.8651f, 0.8677f));
    add_point (0.5019f, make_vec3f (0.8673f, 0.8645f, 0.8626f));
    add_point (0.5058f, make_vec3f (0.8710f, 0.8627f, 0.8571f));
    add_point (0.5098f, make_vec3f (0.8747f, 0.8609f, 0.8515f));
    add_point (0.5137f, make_vec3f (0.8783f, 0.8589f, 0.8459f));
    add_point (0.5176f, make_vec3f (0.8818f, 0.8569f, 0.8403f));
    add_point (0.5215f, make_vec3f (0.8852f, 0.8548f, 0.8347f));
    add_point (0.5254f, make_vec3f (0.8885f, 0.8526f, 0.8290f));
    add_point (0.5294f, make_vec3f (0.8918f, 0.8504f, 0.8233f));
    add_point (0.5333f, make_vec3f (0.8949f, 0.8480f, 0.8176f));
    add_point (0.5372f, make_vec3f (0.8980f, 0.8456f, 0.8119f));
    add_point (0.5411f, make_vec3f (0.9010f, 0.8431f, 0.8061f));
    add_point (0.5450f, make_vec3f (0.9040f, 0.8405f, 0.8003f));
    add_point (0.5490f, make_vec3f (0.9068f, 0.8378f, 0.7944f));
    add_point (0.5529f, make_vec3f (0.9096f, 0.8351f, 0.7886f));
    add_point (0.5568f, make_vec3f (0.9123f, 0.8322f, 0.7827f));
    add_point (0.5607f, make_vec3f (0.9149f, 0.8293f, 0.7768f));
    add_point (0.5647f, make_vec3f (0.9174f, 0.8263f, 0.7709f));
    add_point (0.5686f, make_vec3f (0.9198f, 0.8233f, 0.7649f));
    add_point (0.5725f, make_vec3f (0.9222f, 0.8201f, 0.7590f));
    add_point (0.5764f, make_vec3f (0.9245f, 0.8169f, 0.7530f));
    add_point (0.5803f, make_vec3f (0.9266f, 0.8136f, 0.7470f));
    add_point (0.5843f, make_vec3f (0.9288f, 0.8103f, 0.7410f));
    add_point (0.5882f, make_vec3f (0.9308f, 0.8068f, 0.7349f));
    add_point (0.5921f, make_vec3f (0.9327f, 0.8033f, 0.7289f));
    add_point (0.5960f, make_vec3f (0.9346f, 0.7997f, 0.7228f));
    add_point (0.6f, make_vec3f (0.9363f, 0.7960f, 0.7167f));
    add_point (0.6039f, make_vec3f (0.9380f, 0.7923f, 0.7106f));
    add_point (0.6078f, make_vec3f (0.9396f, 0.7884f, 0.7045f));
    add_point (0.6117f, make_vec3f (0.9412f, 0.7845f, 0.6984f));
    add_point (0.6156f, make_vec3f (0.9426f, 0.7806f, 0.6923f));
    add_point (0.6196f, make_vec3f (0.9439f, 0.7765f, 0.6861f));
    add_point (0.6235f, make_vec3f (0.9452f, 0.7724f, 0.6800f));
    add_point (0.6274f, make_vec3f (0.9464f, 0.7682f, 0.6738f));
    add_point (0.6313f, make_vec3f (0.9475f, 0.7640f, 0.6677f));
    add_point (0.6352f, make_vec3f (0.9485f, 0.7596f, 0.6615f));
    add_point (0.6392f, make_vec3f (0.9495f, 0.7552f, 0.6553f));
    add_point (0.6431f, make_vec3f (0.9503f, 0.7508f, 0.6491f));
    add_point (0.6470f, make_vec3f (0.9511f, 0.7462f, 0.6429f));
    add_point (0.6509f, make_vec3f (0.9517f, 0.7416f, 0.6368f));
    add_point (0.6549f, make_vec3f (0.9523f, 0.7369f, 0.6306f));
    add_point (0.6588f, make_vec3f (0.9529f, 0.7322f, 0.6244f));
    add_point (0.6627f, make_vec3f (0.9533f, 0.7274f, 0.6182f));
    add_point (0.6666f, make_vec3f (0.9536f, 0.7225f, 0.6120f));
    add_point (0.6705f, make_vec3f (0.9539f, 0.7176f, 0.6058f));
    add_point (0.6745f, make_vec3f (0.9541f, 0.7126f, 0.5996f));
    add_point (0.6784f, make_vec3f (0.9542f, 0.7075f, 0.5934f));
    add_point (0.6823f, make_vec3f (0.9542f, 0.7023f, 0.5873f));
    add_point (0.6862f, make_vec3f (0.9541f, 0.6971f, 0.5811f));
    add_point (0.6901f, make_vec3f (0.9539f, 0.6919f, 0.5749f));
    add_point (0.6941f, make_vec3f (0.9537f, 0.6865f, 0.5687f));
    add_point (0.6980f, make_vec3f (0.9534f, 0.6811f, 0.5626f));
    add_point (0.7019f, make_vec3f (0.9529f, 0.6757f, 0.5564f));
    add_point (0.7058f, make_vec3f (0.9524f, 0.6702f, 0.5503f));
    add_point (0.7098f, make_vec3f (0.9519f, 0.6646f, 0.5441f));
    add_point (0.7137f, make_vec3f (0.9512f, 0.6589f, 0.5380f));
    add_point (0.7176f, make_vec3f (0.9505f, 0.6532f, 0.5319f));
    add_point (0.7215f, make_vec3f (0.9496f, 0.6475f, 0.5258f));
    add_point (0.7254f, make_vec3f (0.9487f, 0.6416f, 0.5197f));
    add_point (0.7294f, make_vec3f (0.9477f, 0.6358f, 0.5136f));
    add_point (0.7333f, make_vec3f (0.9466f, 0.6298f, 0.5075f));
    add_point (0.7372f, make_vec3f (0.9455f, 0.6238f, 0.5015f));
    add_point (0.7411f, make_vec3f (0.9442f, 0.6178f, 0.4954f));
    add_point (0.7450f, make_vec3f (0.9429f, 0.6117f, 0.4894f));
    add_point (0.7490f, make_vec3f (0.9415f, 0.6055f, 0.4834f));
    add_point (0.7529f, make_vec3f (0.9400f, 0.5993f, 0.4774f));
    add_point (0.7568f, make_vec3f (0.9384f, 0.5930f, 0.4714f));
    add_point (0.7607f, make_vec3f (0.9368f, 0.5866f, 0.4654f));
    add_point (0.7647f, make_vec3f (0.9350f, 0.5802f, 0.4595f));
    add_point (0.7686f, make_vec3f (0.9332f, 0.5738f, 0.4536f));
    add_point (0.7725f, make_vec3f (0.9313f, 0.5673f, 0.4477f));
    add_point (0.7764f, make_vec3f (0.9293f, 0.5607f, 0.4418f));
    add_point (0.7803f, make_vec3f (0.9273f, 0.5541f, 0.4359f));
    add_point (0.7843f, make_vec3f (0.9251f, 0.5475f, 0.4300f));
    add_point (0.7882f, make_vec3f (0.9229f, 0.5407f, 0.4242f));
    add_point (0.7921f, make_vec3f (0.9206f, 0.5340f, 0.4184f));
    add_point (0.7960f, make_vec3f (0.9182f, 0.5271f, 0.4126f));
    add_point (0.8f, make_vec3f (0.9158f, 0.5203f, 0.4069f));
    add_point (0.8039f, make_vec3f (0.9132f, 0.5133f, 0.4011f));
    add_point (0.8078f, make_vec3f (0.9106f, 0.5063f, 0.3954f));
    add_point (0.8117f, make_vec3f (0.9079f, 0.4993f, 0.3897f));
    add_point (0.8156f, make_vec3f (0.9052f, 0.4922f, 0.3841f));
    add_point (0.8196f, make_vec3f (0.9023f, 0.4851f, 0.3784f));
    add_point (0.8235f, make_vec3f (0.8994f, 0.4779f, 0.3728f));
    add_point (0.8274f, make_vec3f (0.8964f, 0.4706f, 0.3672f));
    add_point (0.8313f, make_vec3f (0.8933f, 0.4633f, 0.3617f));
    add_point (0.8352f, make_vec3f (0.8901f, 0.4559f, 0.3561f));
    add_point (0.8392f, make_vec3f (0.8869f, 0.4485f, 0.3506f));
    add_point (0.8431f, make_vec3f (0.8836f, 0.4410f, 0.3452f));
    add_point (0.8470f, make_vec3f (0.8802f, 0.4335f, 0.3397f));
    add_point (0.8509f, make_vec3f (0.8767f, 0.4259f, 0.3343f));
    add_point (0.8549f, make_vec3f (0.8732f, 0.4183f, 0.3289f));
    add_point (0.8588f, make_vec3f (0.8696f, 0.4106f, 0.3236f));
    add_point (0.8627f, make_vec3f (0.8659f, 0.4028f, 0.3183f));
    add_point (0.8666f, make_vec3f (0.8622f, 0.3950f, 0.3130f));
    add_point (0.8705f, make_vec3f (0.8583f, 0.3871f, 0.3077f));
    add_point (0.8745f, make_vec3f (0.8544f, 0.3792f, 0.3025f));
    add_point (0.8784f, make_vec3f (0.8505f, 0.3712f, 0.2973f));
    add_point (0.8823f, make_vec3f (0.8464f, 0.3631f, 0.2921f));
    add_point (0.8862f, make_vec3f (0.8423f, 0.3549f, 0.2870f));
    add_point (0.8901f, make_vec3f (0.8381f, 0.3467f, 0.2819f));
    add_point (0.8941f, make_vec3f (0.8339f, 0.3384f, 0.2768f));
    add_point (0.8980f, make_vec3f (0.8295f, 0.3300f, 0.2718f));
    add_point (0.9019f, make_vec3f (0.8251f, 0.3215f, 0.2668f));
    add_point (0.9058f, make_vec3f (0.8207f, 0.3129f, 0.2619f));
    add_point (0.9098f, make_vec3f (0.8162f, 0.3043f, 0.2570f));
    add_point (0.9137f, make_vec3f (0.8116f, 0.2955f, 0.2521f));
    add_point (0.9176f, make_vec3f (0.8069f, 0.2866f, 0.2472f));
    add_point (0.9215f, make_vec3f (0.8022f, 0.2776f, 0.2424f));
    add_point (0.9254f, make_vec3f (0.7974f, 0.2685f, 0.2377f));
    add_point (0.9294f, make_vec3f (0.7925f, 0.2592f, 0.2329f));
    add_point (0.9333f, make_vec3f (0.7876f, 0.2498f, 0.2282f));
    add_point (0.9372f, make_vec3f (0.7826f, 0.2402f, 0.2236f));
    add_point (0.9411f, make_vec3f (0.7775f, 0.2304f, 0.2190f));
    add_point (0.9450f, make_vec3f (0.7724f, 0.2204f, 0.2144f));
    add_point (0.9490f, make_vec3f (0.7672f, 0.2102f, 0.2098f));
    add_point (0.9529f, make_vec3f (0.7620f, 0.1997f, 0.2053f));
    add_point (0.9568f, make_vec3f (0.7567f, 0.1889f, 0.2009f));
    add_point (0.9607f, make_vec3f (0.7514f, 0.1777f, 0.1965f));
    add_point (0.9647f, make_vec3f (0.7459f, 0.1662f, 0.1921f));
    add_point (0.9686f, make_vec3f (0.7405f, 0.1541f, 0.1877f));
    add_point (0.9725f, make_vec3f (0.7349f, 0.1414f, 0.1834f));
    add_point (0.9764f, make_vec3f (0.7293f, 0.1279f, 0.1792f));
    add_point (0.9803f, make_vec3f (0.7237f, 0.1134f, 0.1750f));
    add_point (0.9843f, make_vec3f (0.7180f, 0.0975f, 0.1708f));
    add_point (0.9882f, make_vec3f (0.7122f, 0.0796f, 0.1667f));
    add_point (0.9921f, make_vec3f (0.7064f, 0.0585f, 0.1626f));
    add_point (0.9960f, make_vec3f (0.7005f, 0.0315f, 0.1585f));
    add_point (1.0f, make_vec3f (0.6946f, 0.0029f, 0.1545f));
  }
  else if (name == "temperature")
  {
    add_point (0.05f, make_vec3f (0.f, 0.f, 1.f));
    add_point (0.35f, make_vec3f (0.f, 1.f, 1.f));
    add_point (0.50f, make_vec3f (1.f, 1.f, 1.f));
    add_point (0.65f, make_vec3f (1.f, 1.f, 0.f));
    add_point (0.95f, make_vec3f (1.f, 0.f, 0.f));
  }
  else if (name == "rainbow")
  {
    // I really want to delete this. If users want to make a crap
    // color map, let them build it themselves.
    add_point (0.00f, make_vec3f (0.f, 0.f, 1.f));
    add_point (0.20f, make_vec3f (0.f, 1.f, 1.f));
    add_point (0.45f, make_vec3f (0.f, 1.f, 0.f));
    add_point (0.55f, make_vec3f (.7f, 1.f, 0.f));
    add_point (0.6f, make_vec3f (1.f, 1.f, 0.f));
    add_point (0.75f, make_vec3f (1.f, .5f, 0.f));
    add_point (0.9f, make_vec3f (1.f, 0.f, 0.f));
    add_point (0.98f, make_vec3f (1.f, 0.f, .5F));
    add_point (1.0f, make_vec3f (1.f, 0.f, 1.f));
  }
  else if (name == "levels")
  {
    add_point (0.0f, make_vec3f (0.f, 0.f, 1.f));
    add_point (0.2f, make_vec3f (0.f, 0.f, 1.f));
    add_point (0.2f, make_vec3f (0.f, 1.f, 1.f));
    add_point (0.4f, make_vec3f (0.f, 1.f, 1.f));
    add_point (0.4f, make_vec3f (0.f, 1.f, 0.f));
    add_point (0.6f, make_vec3f (0.f, 1.f, 0.f));
    add_point (0.6f, make_vec3f (1.f, 1.f, 0.f));
    add_point (0.8f, make_vec3f (1.f, 1.f, 0.f));
    add_point (0.8f, make_vec3f (1.f, 0.f, 0.f));
    add_point (1.0f, make_vec3f (1.f, 0.f, 0.f));
  }
  else if (name == "dense" || name == "sharp")
  {
    // I'm not fond of this color map either.
    this->m_internals->m_smooth = (name == "dense") ? true : false;
    add_point (0.0f, make_vec3f (0.26f, 0.22f, 0.92f));
    add_point (0.1f, make_vec3f (0.00f, 0.00f, 0.52f));
    add_point (0.2f, make_vec3f (0.00f, 1.00f, 1.00f));
    add_point (0.3f, make_vec3f (0.00f, 0.50f, 0.00f));
    add_point (0.4f, make_vec3f (1.00f, 1.00f, 0.00f));
    add_point (0.5f, make_vec3f (0.60f, 0.47f, 0.00f));
    add_point (0.6f, make_vec3f (1.00f, 0.47f, 0.00f));
    add_point (0.7f, make_vec3f (0.61f, 0.18f, 0.00f));
    add_point (0.8f, make_vec3f (1.00f, 0.03f, 0.17f));
    add_point (0.9f, make_vec3f (0.63f, 0.12f, 0.34f));
    add_point (1.0f, make_vec3f (1.00f, 0.40f, 1.00f));
  }
  else if (name == "thermal")
  {
    add_point (0.0f, make_vec3f (0.30f, 0.00f, 0.00f));
    add_point (0.25f, make_vec3f (1.00f, 0.00f, 0.00f));
    add_point (0.50f, make_vec3f (1.00f, 1.00f, 0.00f));
    add_point (0.55f, make_vec3f (0.80f, 0.55f, 0.20f));
    add_point (0.60f, make_vec3f (0.60f, 0.37f, 0.40f));
    add_point (0.65f, make_vec3f (0.40f, 0.22f, 0.60f));
    add_point (0.75f, make_vec3f (0.00f, 0.00f, 1.00f));
    add_point (1.00f, make_vec3f (1.00f, 1.00f, 1.00f));
  }
  // The following five tables are perceeptually linearized colortables
  // (4 rainbow, one heatmap) from BSD-licensed code by Matteo Niccoli.
  // See: http://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
  else if (name == "IsoL")
  {
    float32 n = 5;
    add_point (0.f / n, make_vec3f (0.9102f, 0.2236f, 0.8997f));
    add_point (1.f / n, make_vec3f (0.4027f, 0.3711f, 1.0000f));
    add_point (2.f / n, make_vec3f (0.0422f, 0.5904f, 0.5899f));
    add_point (3.f / n, make_vec3f (0.0386f, 0.6206f, 0.0201f));
    add_point (4.f / n, make_vec3f (0.5441f, 0.5428f, 0.0110f));
    add_point (5.f / n, make_vec3f (1.0000f, 0.2288f, 0.1631f));
  }
  else if (name == "CubicL")
  {
    float32 n = 15;
    add_point (0.f / n, make_vec3f (0.4706f, 0.0000f, 0.5216f));
    add_point (1.f / n, make_vec3f (0.5137f, 0.0527f, 0.7096f));
    add_point (2.f / n, make_vec3f (0.4942f, 0.2507f, 0.8781f));
    add_point (3.f / n, make_vec3f (0.4296f, 0.3858f, 0.9922f));
    add_point (4.f / n, make_vec3f (0.3691f, 0.5172f, 0.9495f));
    add_point (5.f / n, make_vec3f (0.2963f, 0.6191f, 0.8515f));
    add_point (6.f / n, make_vec3f (0.2199f, 0.7134f, 0.7225f));
    add_point (7.f / n, make_vec3f (0.2643f, 0.7836f, 0.5756f));
    add_point (8.f / n, make_vec3f (0.3094f, 0.8388f, 0.4248f));
    add_point (9.f / n, make_vec3f (0.3623f, 0.8917f, 0.2858f));
    add_point (10.f / n, make_vec3f (0.5200f, 0.9210f, 0.3137f));
    add_point (11.f / n, make_vec3f (0.6800f, 0.9255f, 0.3386f));
    add_point (12.f / n, make_vec3f (0.8000f, 0.9255f, 0.3529f));
    add_point (13.f / n, make_vec3f (0.8706f, 0.8549f, 0.3608f));
    add_point (14.f / n, make_vec3f (0.9514f, 0.7466f, 0.3686f));
    add_point (15.f / n, make_vec3f (0.9765f, 0.5887f, 0.3569f));
  }
  else if (name == "CubicYF")
  {
    float32 n = 15;
    add_point (0.f / n, make_vec3f (0.5151f, 0.0482f, 0.6697f));
    add_point (1.f / n, make_vec3f (0.5199f, 0.1762f, 0.8083f));
    add_point (2.f / n, make_vec3f (0.4884f, 0.2912f, 0.9234f));
    add_point (3.f / n, make_vec3f (0.4297f, 0.3855f, 0.9921f));
    add_point (4.f / n, make_vec3f (0.3893f, 0.4792f, 0.9775f));
    add_point (5.f / n, make_vec3f (0.3337f, 0.5650f, 0.9056f));
    add_point (6.f / n, make_vec3f (0.2795f, 0.6419f, 0.8287f));
    add_point (7.f / n, make_vec3f (0.2210f, 0.7123f, 0.7258f));
    add_point (8.f / n, make_vec3f (0.2468f, 0.7612f, 0.6248f));
    add_point (9.f / n, make_vec3f (0.2833f, 0.8125f, 0.5069f));
    add_point (10.f / n, make_vec3f (0.3198f, 0.8492f, 0.3956f));
    add_point (11.f / n, make_vec3f (0.3602f, 0.8896f, 0.2919f));
    add_point (12.f / n, make_vec3f (0.4568f, 0.9136f, 0.3018f));
    add_point (13.f / n, make_vec3f (0.6033f, 0.9255f, 0.3295f));
    add_point (14.f / n, make_vec3f (0.7066f, 0.9255f, 0.3414f));
    add_point (15.f / n, make_vec3f (0.8000f, 0.9255f, 0.3529f));
  }
  else if (name == "LinearL")
  {
    float32 n = 15;
    add_point (0.f / n, make_vec3f (0.0143f, 0.0143f, 0.0143f));
    add_point (1.f / n, make_vec3f (0.1413f, 0.0555f, 0.1256f));
    add_point (2.f / n, make_vec3f (0.1761f, 0.0911f, 0.2782f));
    add_point (3.f / n, make_vec3f (0.1710f, 0.1314f, 0.4540f));
    add_point (4.f / n, make_vec3f (0.1074f, 0.2234f, 0.4984f));
    add_point (5.f / n, make_vec3f (0.0686f, 0.3044f, 0.5068f));
    add_point (6.f / n, make_vec3f (0.0008f, 0.3927f, 0.4267f));
    add_point (7.f / n, make_vec3f (0.0000f, 0.4763f, 0.3464f));
    add_point (8.f / n, make_vec3f (0.0000f, 0.5565f, 0.2469f));
    add_point (9.f / n, make_vec3f (0.0000f, 0.6381f, 0.1638f));
    add_point (10.f / n, make_vec3f (0.2167f, 0.6966f, 0.0000f));
    add_point (11.f / n, make_vec3f (0.3898f, 0.7563f, 0.0000f));
    add_point (12.f / n, make_vec3f (0.6912f, 0.7795f, 0.0000f));
    add_point (13.f / n, make_vec3f (0.8548f, 0.8041f, 0.4555f));
    add_point (14.f / n, make_vec3f (0.9712f, 0.8429f, 0.7287f));
    add_point (15.f / n, make_vec3f (0.9692f, 0.9273f, 0.8961f));
  }
  else if (name == "LinLhot")
  {
    float32 n = 15;
    add_point (0.f / n, make_vec3f (0.0225f, 0.0121f, 0.0121f));
    add_point (1.f / n, make_vec3f (0.1927f, 0.0225f, 0.0311f));
    add_point (2.f / n, make_vec3f (0.3243f, 0.0106f, 0.0000f));
    add_point (3.f / n, make_vec3f (0.4463f, 0.0000f, 0.0091f));
    add_point (4.f / n, make_vec3f (0.5706f, 0.0000f, 0.0737f));
    add_point (5.f / n, make_vec3f (0.6969f, 0.0000f, 0.1337f));
    add_point (6.f / n, make_vec3f (0.8213f, 0.0000f, 0.1792f));
    add_point (7.f / n, make_vec3f (0.8636f, 0.0000f, 0.0565f));
    add_point (8.f / n, make_vec3f (0.8821f, 0.2555f, 0.0000f));
    add_point (9.f / n, make_vec3f (0.8720f, 0.4182f, 0.0000f));
    add_point (10.f / n, make_vec3f (0.8424f, 0.5552f, 0.0000f));
    add_point (11.f / n, make_vec3f (0.8031f, 0.6776f, 0.0000f));
    add_point (12.f / n, make_vec3f (0.7659f, 0.7870f, 0.0000f));
    add_point (13.f / n, make_vec3f (0.8170f, 0.8296f, 0.0000f));
    add_point (14.f / n, make_vec3f (0.8853f, 0.8896f, 0.4113f));
    add_point (15.f / n, make_vec3f (0.9481f, 0.9486f, 0.7165f));
  }
  // ColorBrewer tables here.  (See LICENSE.txt)
  else if (name == "PuRd")
  {
    add_point (0.0000f, make_vec3f (0.9686f, 0.9569f, 0.9765f));
    add_point (0.1250f, make_vec3f (0.9059f, 0.8824f, 0.9373f));
    add_point (0.2500f, make_vec3f (0.8314f, 0.7255f, 0.8549f));
    add_point (0.3750f, make_vec3f (0.7882f, 0.5804f, 0.7804f));
    add_point (0.5000f, make_vec3f (0.8745f, 0.3961f, 0.6902f));
    add_point (0.6250f, make_vec3f (0.9059f, 0.1608f, 0.5412f));
    add_point (0.7500f, make_vec3f (0.8078f, 0.0706f, 0.3373f));
    add_point (0.8750f, make_vec3f (0.5961f, 0.0000f, 0.2627f));
    add_point (1.0000f, make_vec3f (0.4039f, 0.0000f, 0.1216f));
  }
  else if (name == "Accent")
  {
    add_point (0.0000f, make_vec3f (0.4980f, 0.7882f, 0.4980f));
    add_point (0.1429f, make_vec3f (0.7451f, 0.6824f, 0.8314f));
    add_point (0.2857f, make_vec3f (0.9922f, 0.7529f, 0.5255f));
    add_point (0.4286f, make_vec3f (1.0000f, 1.0000f, 0.6000f));
    add_point (0.5714f, make_vec3f (0.2196f, 0.4235f, 0.6902f));
    add_point (0.7143f, make_vec3f (0.9412f, 0.0078f, 0.4980f));
    add_point (0.8571f, make_vec3f (0.7490f, 0.3569f, 0.0902f));
    add_point (1.0000f, make_vec3f (0.4000f, 0.4000f, 0.4000f));
  }
  else if (name == "Blues")
  {
    add_point (0.0000f, make_vec3f (0.9686f, 0.9843f, 1.0000f));
    add_point (0.1250f, make_vec3f (0.8706f, 0.9216f, 0.9686f));
    add_point (0.2500f, make_vec3f (0.7765f, 0.8588f, 0.9373f));
    add_point (0.3750f, make_vec3f (0.6196f, 0.7922f, 0.8824f));
    add_point (0.5000f, make_vec3f (0.4196f, 0.6824f, 0.8392f));
    add_point (0.6250f, make_vec3f (0.2588f, 0.5725f, 0.7765f));
    add_point (0.7500f, make_vec3f (0.1294f, 0.4431f, 0.7098f));
    add_point (0.8750f, make_vec3f (0.0314f, 0.3176f, 0.6118f));
    add_point (1.0000f, make_vec3f (0.0314f, 0.1882f, 0.4196f));
  }
  else if (name == "BrBG")
  {
    add_point (0.0000f, make_vec3f (0.3294f, 0.1882f, 0.0196f));
    add_point (0.1000f, make_vec3f (0.5490f, 0.3176f, 0.0392f));
    add_point (0.2000f, make_vec3f (0.7490f, 0.5059f, 0.1765f));
    add_point (0.3000f, make_vec3f (0.8745f, 0.7608f, 0.4902f));
    add_point (0.4000f, make_vec3f (0.9647f, 0.9098f, 0.7647f));
    add_point (0.5000f, make_vec3f (0.9608f, 0.9608f, 0.9608f));
    add_point (0.6000f, make_vec3f (0.7804f, 0.9176f, 0.8980f));
    add_point (0.7000f, make_vec3f (0.5020f, 0.8039f, 0.7569f));
    add_point (0.8000f, make_vec3f (0.2078f, 0.5922f, 0.5608f));
    add_point (0.9000f, make_vec3f (0.0039f, 0.4000f, 0.3686f));
    add_point (1.0000f, make_vec3f (0.0000f, 0.2353f, 0.1882f));
  }
  else if (name == "BuGn")
  {
    add_point (0.0000f, make_vec3f (0.9686f, 0.9882f, 0.9922f));
    add_point (0.1250f, make_vec3f (0.8980f, 0.9608f, 0.9765f));
    add_point (0.2500f, make_vec3f (0.8000f, 0.9255f, 0.9020f));
    add_point (0.3750f, make_vec3f (0.6000f, 0.8471f, 0.7882f));
    add_point (0.5000f, make_vec3f (0.4000f, 0.7608f, 0.6431f));
    add_point (0.6250f, make_vec3f (0.2549f, 0.6824f, 0.4627f));
    add_point (0.7500f, make_vec3f (0.1373f, 0.5451f, 0.2706f));
    add_point (0.8750f, make_vec3f (0.0000f, 0.4275f, 0.1725f));
    add_point (1.0000f, make_vec3f (0.0000f, 0.2667f, 0.1059f));
  }
  else if (name == "BuPu")
  {
    add_point (0.0000f, make_vec3f (0.9686f, 0.9882f, 0.9922f));
    add_point (0.1250f, make_vec3f (0.8784f, 0.9255f, 0.9569f));
    add_point (0.2500f, make_vec3f (0.7490f, 0.8275f, 0.9020f));
    add_point (0.3750f, make_vec3f (0.6196f, 0.7373f, 0.8549f));
    add_point (0.5000f, make_vec3f (0.5490f, 0.5882f, 0.7765f));
    add_point (0.6250f, make_vec3f (0.5490f, 0.4196f, 0.6941f));
    add_point (0.7500f, make_vec3f (0.5333f, 0.2549f, 0.6157f));
    add_point (0.8750f, make_vec3f (0.5059f, 0.0588f, 0.4863f));
    add_point (1.0000f, make_vec3f (0.3020f, 0.0000f, 0.2941f));
  }
  else if (name == "Dark2")
  {
    add_point (0.0000f, make_vec3f (0.1059f, 0.6196f, 0.4667f));
    add_point (0.1429f, make_vec3f (0.8510f, 0.3725f, 0.0078f));
    add_point (0.2857f, make_vec3f (0.4588f, 0.4392f, 0.7020f));
    add_point (0.4286f, make_vec3f (0.9059f, 0.1608f, 0.5412f));
    add_point (0.5714f, make_vec3f (0.4000f, 0.6510f, 0.1176f));
    add_point (0.7143f, make_vec3f (0.9020f, 0.6706f, 0.0078f));
    add_point (0.8571f, make_vec3f (0.6510f, 0.4627f, 0.1137f));
    add_point (1.0000f, make_vec3f (0.4000f, 0.4000f, 0.4000f));
  }
  else if (name == "GnBu")
  {
    add_point (0.0000f, make_vec3f (0.9686f, 0.9882f, 0.9412f));
    add_point (0.1250f, make_vec3f (0.8784f, 0.9529f, 0.8588f));
    add_point (0.2500f, make_vec3f (0.8000f, 0.9216f, 0.7725f));
    add_point (0.3750f, make_vec3f (0.6588f, 0.8667f, 0.7098f));
    add_point (0.5000f, make_vec3f (0.4824f, 0.8000f, 0.7686f));
    add_point (0.6250f, make_vec3f (0.3059f, 0.7020f, 0.8275f));
    add_point (0.7500f, make_vec3f (0.1686f, 0.5490f, 0.7451f));
    add_point (0.8750f, make_vec3f (0.0314f, 0.4078f, 0.6745f));
    add_point (1.0000f, make_vec3f (0.0314f, 0.2510f, 0.5059f));
  }
  else if (name == "Greens")
  {
    add_point (0.0000f, make_vec3f (0.9686f, 0.9882f, 0.9608f));
    add_point (0.1250f, make_vec3f (0.8980f, 0.9608f, 0.8784f));
    add_point (0.2500f, make_vec3f (0.7804f, 0.9137f, 0.7529f));
    add_point (0.3750f, make_vec3f (0.6314f, 0.8510f, 0.6078f));
    add_point (0.5000f, make_vec3f (0.4549f, 0.7686f, 0.4627f));
    add_point (0.6250f, make_vec3f (0.2549f, 0.6706f, 0.3647f));
    add_point (0.7500f, make_vec3f (0.1373f, 0.5451f, 0.2706f));
    add_point (0.8750f, make_vec3f (0.0000f, 0.4275f, 0.1725f));
    add_point (1.0000f, make_vec3f (0.0000f, 0.2667f, 0.1059f));
  }
  else if (name == "Greys")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 1.0000f, 1.0000f));
    add_point (0.1250f, make_vec3f (0.9412f, 0.9412f, 0.9412f));
    add_point (0.2500f, make_vec3f (0.8510f, 0.8510f, 0.8510f));
    add_point (0.3750f, make_vec3f (0.7412f, 0.7412f, 0.7412f));
    add_point (0.5000f, make_vec3f (0.5882f, 0.5882f, 0.5882f));
    add_point (0.6250f, make_vec3f (0.4510f, 0.4510f, 0.4510f));
    add_point (0.7500f, make_vec3f (0.3216f, 0.3216f, 0.3216f));
    add_point (0.8750f, make_vec3f (0.1451f, 0.1451f, 0.1451f));
    add_point (1.0000f, make_vec3f (0.0000f, 0.0000f, 0.0000f));
  }
  else if (name == "Oranges")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 0.9608f, 0.9216f));
    add_point (0.1250f, make_vec3f (0.9961f, 0.9020f, 0.8078f));
    add_point (0.2500f, make_vec3f (0.9922f, 0.8157f, 0.6353f));
    add_point (0.3750f, make_vec3f (0.9922f, 0.6824f, 0.4196f));
    add_point (0.5000f, make_vec3f (0.9922f, 0.5529f, 0.2353f));
    add_point (0.6250f, make_vec3f (0.9451f, 0.4118f, 0.0745f));
    add_point (0.7500f, make_vec3f (0.8510f, 0.2824f, 0.0039f));
    add_point (0.8750f, make_vec3f (0.6510f, 0.2118f, 0.0118f));
    add_point (1.0000f, make_vec3f (0.4980f, 0.1529f, 0.0157f));
  }
  else if (name == "OrRd")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 0.9686f, 0.9255f));
    add_point (0.1250f, make_vec3f (0.9961f, 0.9098f, 0.7843f));
    add_point (0.2500f, make_vec3f (0.9922f, 0.8314f, 0.6196f));
    add_point (0.3750f, make_vec3f (0.9922f, 0.7333f, 0.5176f));
    add_point (0.5000f, make_vec3f (0.9882f, 0.5529f, 0.3490f));
    add_point (0.6250f, make_vec3f (0.9373f, 0.3961f, 0.2824f));
    add_point (0.7500f, make_vec3f (0.8431f, 0.1882f, 0.1216f));
    add_point (0.8750f, make_vec3f (0.7020f, 0.0000f, 0.0000f));
    add_point (1.0000f, make_vec3f (0.4980f, 0.0000f, 0.0000f));
  }
  else if (name == "Paired")
  {
    add_point (0.0000f, make_vec3f (0.6510f, 0.8078f, 0.8902f));
    add_point (0.0909f, make_vec3f (0.1216f, 0.4706f, 0.7059f));
    add_point (0.1818f, make_vec3f (0.6980f, 0.8745f, 0.5412f));
    add_point (0.2727f, make_vec3f (0.2000f, 0.6275f, 0.1725f));
    add_point (0.3636f, make_vec3f (0.9843f, 0.6039f, 0.6000f));
    add_point (0.4545f, make_vec3f (0.8902f, 0.1020f, 0.1098f));
    add_point (0.5455f, make_vec3f (0.9922f, 0.7490f, 0.4353f));
    add_point (0.6364f, make_vec3f (1.0000f, 0.4980f, 0.0000f));
    add_point (0.7273f, make_vec3f (0.7922f, 0.6980f, 0.8392f));
    add_point (0.8182f, make_vec3f (0.4157f, 0.2392f, 0.6039f));
    add_point (0.9091f, make_vec3f (1.0000f, 1.0000f, 0.6000f));
    add_point (1.0000f, make_vec3f (0.6941f, 0.3490f, 0.1569f));
  }
  else if (name == "Pastel1")
  {
    add_point (0.0000f, make_vec3f (0.9843f, 0.7059f, 0.6824f));
    add_point (0.1250f, make_vec3f (0.7020f, 0.8039f, 0.8902f));
    add_point (0.2500f, make_vec3f (0.8000f, 0.9216f, 0.7725f));
    add_point (0.3750f, make_vec3f (0.8706f, 0.7961f, 0.8941f));
    add_point (0.5000f, make_vec3f (0.9961f, 0.8510f, 0.6510f));
    add_point (0.6250f, make_vec3f (1.0000f, 1.0000f, 0.8000f));
    add_point (0.7500f, make_vec3f (0.8980f, 0.8471f, 0.7412f));
    add_point (0.8750f, make_vec3f (0.9922f, 0.8549f, 0.9255f));
    add_point (1.0000f, make_vec3f (0.9490f, 0.9490f, 0.9490f));
  }
  else if (name == "Pastel2")
  {
    add_point (0.0000f, make_vec3f (0.7020f, 0.8863f, 0.8039f));
    add_point (0.1429f, make_vec3f (0.9922f, 0.8039f, 0.6745f));
    add_point (0.2857f, make_vec3f (0.7961f, 0.8353f, 0.9098f));
    add_point (0.4286f, make_vec3f (0.9569f, 0.7922f, 0.8941f));
    add_point (0.5714f, make_vec3f (0.9020f, 0.9608f, 0.7882f));
    add_point (0.7143f, make_vec3f (1.0000f, 0.9490f, 0.6824f));
    add_point (0.8571f, make_vec3f (0.9451f, 0.8863f, 0.8000f));
    add_point (1.0000f, make_vec3f (0.8000f, 0.8000f, 0.8000f));
  }
  else if (name == "PiYG")
  {
    add_point (0.0000f, make_vec3f (0.5569f, 0.0039f, 0.3216f));
    add_point (0.1000f, make_vec3f (0.7725f, 0.1059f, 0.4902f));
    add_point (0.2000f, make_vec3f (0.8706f, 0.4667f, 0.6824f));
    add_point (0.3000f, make_vec3f (0.9451f, 0.7137f, 0.8549f));
    add_point (0.4000f, make_vec3f (0.9922f, 0.8784f, 0.9373f));
    add_point (0.5000f, make_vec3f (0.9686f, 0.9686f, 0.9686f));
    add_point (0.6000f, make_vec3f (0.9020f, 0.9608f, 0.8157f));
    add_point (0.7000f, make_vec3f (0.7216f, 0.8824f, 0.5255f));
    add_point (0.8000f, make_vec3f (0.4980f, 0.7373f, 0.2549f));
    add_point (0.9000f, make_vec3f (0.3020f, 0.5725f, 0.1294f));
    add_point (1.0000f, make_vec3f (0.1529f, 0.3922f, 0.0980f));
  }
  else if (name == "PRGn")
  {
    add_point (0.0000f, make_vec3f (0.2510f, 0.0000f, 0.2941f));
    add_point (0.1000f, make_vec3f (0.4627f, 0.1647f, 0.5137f));
    add_point (0.2000f, make_vec3f (0.6000f, 0.4392f, 0.6706f));
    add_point (0.3000f, make_vec3f (0.7608f, 0.6471f, 0.8118f));
    add_point (0.4000f, make_vec3f (0.9059f, 0.8314f, 0.9098f));
    add_point (0.5000f, make_vec3f (0.9686f, 0.9686f, 0.9686f));
    add_point (0.6000f, make_vec3f (0.8510f, 0.9412f, 0.8275f));
    add_point (0.7000f, make_vec3f (0.6510f, 0.8588f, 0.6275f));
    add_point (0.8000f, make_vec3f (0.3529f, 0.6824f, 0.3804f));
    add_point (0.9000f, make_vec3f (0.1059f, 0.4706f, 0.2157f));
    add_point (1.0000f, make_vec3f (0.0000f, 0.2667f, 0.1059f));
  }
  else if (name == "PuBu")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 0.9686f, 0.9843f));
    add_point (0.1250f, make_vec3f (0.9255f, 0.9059f, 0.9490f));
    add_point (0.2500f, make_vec3f (0.8157f, 0.8196f, 0.9020f));
    add_point (0.3750f, make_vec3f (0.6510f, 0.7412f, 0.8588f));
    add_point (0.5000f, make_vec3f (0.4549f, 0.6627f, 0.8118f));
    add_point (0.6250f, make_vec3f (0.2118f, 0.5647f, 0.7529f));
    add_point (0.7500f, make_vec3f (0.0196f, 0.4392f, 0.6902f));
    add_point (0.8750f, make_vec3f (0.0157f, 0.3529f, 0.5529f));
    add_point (1.0000f, make_vec3f (0.0078f, 0.2196f, 0.3451f));
  }
  else if (name == "PuBuGn")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 0.9686f, 0.9843f));
    add_point (0.1250f, make_vec3f (0.9255f, 0.8863f, 0.9412f));
    add_point (0.2500f, make_vec3f (0.8157f, 0.8196f, 0.9020f));
    add_point (0.3750f, make_vec3f (0.6510f, 0.7412f, 0.8588f));
    add_point (0.5000f, make_vec3f (0.4039f, 0.6627f, 0.8118f));
    add_point (0.6250f, make_vec3f (0.2118f, 0.5647f, 0.7529f));
    add_point (0.7500f, make_vec3f (0.0078f, 0.5059f, 0.5412f));
    add_point (0.8750f, make_vec3f (0.0039f, 0.4235f, 0.3490f));
    add_point (1.0000f, make_vec3f (0.0039f, 0.2745f, 0.2118f));
  }
  else if (name == "PuOr")
  {
    add_point (0.0000f, make_vec3f (0.4980f, 0.2314f, 0.0314f));
    add_point (0.1000f, make_vec3f (0.7020f, 0.3451f, 0.0235f));
    add_point (0.2000f, make_vec3f (0.8784f, 0.5098f, 0.0784f));
    add_point (0.3000f, make_vec3f (0.9922f, 0.7216f, 0.3882f));
    add_point (0.4000f, make_vec3f (0.9961f, 0.8784f, 0.7137f));
    add_point (0.5000f, make_vec3f (0.9686f, 0.9686f, 0.9686f));
    add_point (0.6000f, make_vec3f (0.8471f, 0.8549f, 0.9216f));
    add_point (0.7000f, make_vec3f (0.6980f, 0.6706f, 0.8235f));
    add_point (0.8000f, make_vec3f (0.5020f, 0.4510f, 0.6745f));
    add_point (0.9000f, make_vec3f (0.3294f, 0.1529f, 0.5333f));
    add_point (1.0000f, make_vec3f (0.1765f, 0.0000f, 0.2941f));
  }
  else if (name == "PuRd")
  {
    add_point (0.0000f, make_vec3f (0.9686f, 0.9569f, 0.9765f));
    add_point (0.1250f, make_vec3f (0.9059f, 0.8824f, 0.9373f));
    add_point (0.2500f, make_vec3f (0.8314f, 0.7255f, 0.8549f));
    add_point (0.3750f, make_vec3f (0.7882f, 0.5804f, 0.7804f));
    add_point (0.5000f, make_vec3f (0.8745f, 0.3961f, 0.6902f));
    add_point (0.6250f, make_vec3f (0.9059f, 0.1608f, 0.5412f));
    add_point (0.7500f, make_vec3f (0.8078f, 0.0706f, 0.3373f));
    add_point (0.8750f, make_vec3f (0.5961f, 0.0000f, 0.2627f));
    add_point (1.0000f, make_vec3f (0.4039f, 0.0000f, 0.1216f));
  }
  else if (name == "Purples")
  {
    add_point (0.0000f, make_vec3f (0.9882f, 0.9843f, 0.9922f));
    add_point (0.1250f, make_vec3f (0.9373f, 0.9294f, 0.9608f));
    add_point (0.2500f, make_vec3f (0.8549f, 0.8549f, 0.9216f));
    add_point (0.3750f, make_vec3f (0.7373f, 0.7412f, 0.8627f));
    add_point (0.5000f, make_vec3f (0.6196f, 0.6039f, 0.7843f));
    add_point (0.6250f, make_vec3f (0.5020f, 0.4902f, 0.7294f));
    add_point (0.7500f, make_vec3f (0.4157f, 0.3176f, 0.6392f));
    add_point (0.8750f, make_vec3f (0.3294f, 0.1529f, 0.5608f));
    add_point (1.0000f, make_vec3f (0.2471f, 0.0000f, 0.4902f));
  }
  else if (name == "RdBu")
  {
    add_point (0.0000f, make_vec3f (0.4039f, 0.0000f, 0.1216f));
    add_point (0.1000f, make_vec3f (0.6980f, 0.0941f, 0.1686f));
    add_point (0.2000f, make_vec3f (0.8392f, 0.3765f, 0.3020f));
    add_point (0.3000f, make_vec3f (0.9569f, 0.6471f, 0.5098f));
    add_point (0.4000f, make_vec3f (0.9922f, 0.8588f, 0.7804f));
    add_point (0.5000f, make_vec3f (0.9686f, 0.9686f, 0.9686f));
    add_point (0.6000f, make_vec3f (0.8196f, 0.8980f, 0.9412f));
    add_point (0.7000f, make_vec3f (0.5725f, 0.7725f, 0.8706f));
    add_point (0.8000f, make_vec3f (0.2627f, 0.5765f, 0.7647f));
    add_point (0.9000f, make_vec3f (0.1294f, 0.4000f, 0.6745f));
    add_point (1.0000f, make_vec3f (0.0196f, 0.1882f, 0.3804f));
  }
  else if (name == "RdGy")
  {
    add_point (0.0000f, make_vec3f (0.4039f, 0.0000f, 0.1216f));
    add_point (0.1000f, make_vec3f (0.6980f, 0.0941f, 0.1686f));
    add_point (0.2000f, make_vec3f (0.8392f, 0.3765f, 0.3020f));
    add_point (0.3000f, make_vec3f (0.9569f, 0.6471f, 0.5098f));
    add_point (0.4000f, make_vec3f (0.9922f, 0.8588f, 0.7804f));
    add_point (0.5000f, make_vec3f (1.0000f, 1.0000f, 1.0000f));
    add_point (0.6000f, make_vec3f (0.8784f, 0.8784f, 0.8784f));
    add_point (0.7000f, make_vec3f (0.7294f, 0.7294f, 0.7294f));
    add_point (0.8000f, make_vec3f (0.5294f, 0.5294f, 0.5294f));
    add_point (0.9000f, make_vec3f (0.3020f, 0.3020f, 0.3020f));
    add_point (1.0000f, make_vec3f (0.1020f, 0.1020f, 0.1020f));
  }
  else if (name == "RdPu")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 0.9686f, 0.9529f));
    add_point (0.1250f, make_vec3f (0.9922f, 0.8784f, 0.8667f));
    add_point (0.2500f, make_vec3f (0.9882f, 0.7725f, 0.7529f));
    add_point (0.3750f, make_vec3f (0.9804f, 0.6235f, 0.7098f));
    add_point (0.5000f, make_vec3f (0.9686f, 0.4078f, 0.6314f));
    add_point (0.6250f, make_vec3f (0.8667f, 0.2039f, 0.5922f));
    add_point (0.7500f, make_vec3f (0.6824f, 0.0039f, 0.4941f));
    add_point (0.8750f, make_vec3f (0.4784f, 0.0039f, 0.4667f));
    add_point (1.0000f, make_vec3f (0.2863f, 0.0000f, 0.4157f));
  }
  else if (name == "RdYlBu")
  {
    add_point (0.0000f, make_vec3f (0.6471f, 0.0000f, 0.1490f));
    add_point (0.1000f, make_vec3f (0.8431f, 0.1882f, 0.1529f));
    add_point (0.2000f, make_vec3f (0.9569f, 0.4275f, 0.2627f));
    add_point (0.3000f, make_vec3f (0.9922f, 0.6824f, 0.3804f));
    add_point (0.4000f, make_vec3f (0.9961f, 0.8784f, 0.5647f));
    add_point (0.5000f, make_vec3f (1.0000f, 1.0000f, 0.7490f));
    add_point (0.6000f, make_vec3f (0.8784f, 0.9529f, 0.9725f));
    add_point (0.7000f, make_vec3f (0.6706f, 0.8510f, 0.9137f));
    add_point (0.8000f, make_vec3f (0.4549f, 0.6784f, 0.8196f));
    add_point (0.9000f, make_vec3f (0.2706f, 0.4588f, 0.7059f));
    add_point (1.0000f, make_vec3f (0.1922f, 0.2118f, 0.5843f));
  }
  else if (name == "RdYlGn")
  {
    add_point (0.0000f, make_vec3f (0.6471f, 0.0000f, 0.1490f));
    add_point (0.1000f, make_vec3f (0.8431f, 0.1882f, 0.1529f));
    add_point (0.2000f, make_vec3f (0.9569f, 0.4275f, 0.2627f));
    add_point (0.3000f, make_vec3f (0.9922f, 0.6824f, 0.3804f));
    add_point (0.4000f, make_vec3f (0.9961f, 0.8784f, 0.5451f));
    add_point (0.5000f, make_vec3f (1.0000f, 1.0000f, 0.7490f));
    add_point (0.6000f, make_vec3f (0.8510f, 0.9373f, 0.5451f));
    add_point (0.7000f, make_vec3f (0.6510f, 0.8510f, 0.4157f));
    add_point (0.8000f, make_vec3f (0.4000f, 0.7412f, 0.3882f));
    add_point (0.9000f, make_vec3f (0.1020f, 0.5961f, 0.3137f));
    add_point (1.0000f, make_vec3f (0.0000f, 0.4078f, 0.2157f));
  }
  else if (name == "Reds")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 0.9608f, 0.9412f));
    add_point (0.1250f, make_vec3f (0.9961f, 0.8784f, 0.8235f));
    add_point (0.2500f, make_vec3f (0.9882f, 0.7333f, 0.6314f));
    add_point (0.3750f, make_vec3f (0.9882f, 0.5725f, 0.4471f));
    add_point (0.5000f, make_vec3f (0.9843f, 0.4157f, 0.2902f));
    add_point (0.6250f, make_vec3f (0.9373f, 0.2314f, 0.1725f));
    add_point (0.7500f, make_vec3f (0.7961f, 0.0941f, 0.1137f));
    add_point (0.8750f, make_vec3f (0.6471f, 0.0588f, 0.0824f));
    add_point (1.0000f, make_vec3f (0.4039f, 0.0000f, 0.0510f));
  }
  else if (name == "Set1")
  {
    add_point (0.0000f, make_vec3f (0.8941f, 0.1020f, 0.1098f));
    add_point (0.1250f, make_vec3f (0.2157f, 0.4941f, 0.7216f));
    add_point (0.2500f, make_vec3f (0.3020f, 0.6863f, 0.2902f));
    add_point (0.3750f, make_vec3f (0.5961f, 0.3059f, 0.6392f));
    add_point (0.5000f, make_vec3f (1.0000f, 0.4980f, 0.0000f));
    add_point (0.6250f, make_vec3f (1.0000f, 1.0000f, 0.2000f));
    add_point (0.7500f, make_vec3f (0.6510f, 0.3373f, 0.1569f));
    add_point (0.8750f, make_vec3f (0.9686f, 0.5059f, 0.7490f));
    add_point (1.0000f, make_vec3f (0.6000f, 0.6000f, 0.6000f));
  }
  else if (name == "Set2")
  {
    add_point (0.0000f, make_vec3f (0.4000f, 0.7608f, 0.6471f));
    add_point (0.1429f, make_vec3f (0.9882f, 0.5529f, 0.3843f));
    add_point (0.2857f, make_vec3f (0.5529f, 0.6275f, 0.7961f));
    add_point (0.4286f, make_vec3f (0.9059f, 0.5412f, 0.7647f));
    add_point (0.5714f, make_vec3f (0.6510f, 0.8471f, 0.3294f));
    add_point (0.7143f, make_vec3f (1.0000f, 0.8510f, 0.1843f));
    add_point (0.8571f, make_vec3f (0.8980f, 0.7686f, 0.5804f));
    add_point (1.0000f, make_vec3f (0.7020f, 0.7020f, 0.7020f));
  }
  else if (name == "Set3")
  {
    add_point (0.0000f, make_vec3f (0.5529f, 0.8275f, 0.7804f));
    add_point (0.0909f, make_vec3f (1.0000f, 1.0000f, 0.7020f));
    add_point (0.1818f, make_vec3f (0.7451f, 0.7294f, 0.8549f));
    add_point (0.2727f, make_vec3f (0.9843f, 0.5020f, 0.4471f));
    add_point (0.3636f, make_vec3f (0.5020f, 0.6941f, 0.8275f));
    add_point (0.4545f, make_vec3f (0.9922f, 0.7059f, 0.3843f));
    add_point (0.5455f, make_vec3f (0.7020f, 0.8706f, 0.4118f));
    add_point (0.6364f, make_vec3f (0.9882f, 0.8039f, 0.8980f));
    add_point (0.7273f, make_vec3f (0.8510f, 0.8510f, 0.8510f));
    add_point (0.8182f, make_vec3f (0.7373f, 0.5020f, 0.7412f));
    add_point (0.9091f, make_vec3f (0.8000f, 0.9216f, 0.7725f));
    add_point (1.0000f, make_vec3f (1.0000f, 0.9294f, 0.4353f));
  }
  else if (name == "Spectral")
  {
    add_point (0.0000f, make_vec3f (0.6196f, 0.0039f, 0.2588f));
    add_point (0.1000f, make_vec3f (0.8353f, 0.2431f, 0.3098f));
    add_point (0.2000f, make_vec3f (0.9569f, 0.4275f, 0.2627f));
    add_point (0.3000f, make_vec3f (0.9922f, 0.6824f, 0.3804f));
    add_point (0.4000f, make_vec3f (0.9961f, 0.8784f, 0.5451f));
    add_point (0.5000f, make_vec3f (1.0000f, 1.0000f, 0.7490f));
    add_point (0.6000f, make_vec3f (0.9020f, 0.9608f, 0.5961f));
    add_point (0.7000f, make_vec3f (0.6706f, 0.8667f, 0.6431f));
    add_point (0.8000f, make_vec3f (0.4000f, 0.7608f, 0.6471f));
    add_point (0.9000f, make_vec3f (0.1961f, 0.5333f, 0.7412f));
    add_point (1.0000f, make_vec3f (0.3686f, 0.3098f, 0.6353f));
  }
  else if (name == "YlGnBu")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 1.0000f, 0.8510f));
    add_point (0.1250f, make_vec3f (0.9294f, 0.9725f, 0.6941f));
    add_point (0.2500f, make_vec3f (0.7804f, 0.9137f, 0.7059f));
    add_point (0.3750f, make_vec3f (0.4980f, 0.8039f, 0.7333f));
    add_point (0.5000f, make_vec3f (0.2549f, 0.7137f, 0.7686f));
    add_point (0.6250f, make_vec3f (0.1137f, 0.5686f, 0.7529f));
    add_point (0.7500f, make_vec3f (0.1333f, 0.3686f, 0.6588f));
    add_point (0.8750f, make_vec3f (0.1451f, 0.2039f, 0.5804f));
    add_point (1.0000f, make_vec3f (0.0314f, 0.1137f, 0.3451f));
  }
  else if (name == "YlGn")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 1.0000f, 0.8980f));
    add_point (0.1250f, make_vec3f (0.9686f, 0.9882f, 0.7255f));
    add_point (0.2500f, make_vec3f (0.8510f, 0.9412f, 0.6392f));
    add_point (0.3750f, make_vec3f (0.6784f, 0.8667f, 0.5569f));
    add_point (0.5000f, make_vec3f (0.4706f, 0.7765f, 0.4745f));
    add_point (0.6250f, make_vec3f (0.2549f, 0.6706f, 0.3647f));
    add_point (0.7500f, make_vec3f (0.1373f, 0.5176f, 0.2627f));
    add_point (0.8750f, make_vec3f (0.0000f, 0.4078f, 0.2157f));
    add_point (1.0000f, make_vec3f (0.0000f, 0.2706f, 0.1608f));
  }
  else if (name == "YlOrBr")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 1.0000f, 0.8980f));
    add_point (0.1250f, make_vec3f (1.0000f, 0.9686f, 0.7373f));
    add_point (0.2500f, make_vec3f (0.9961f, 0.8902f, 0.5686f));
    add_point (0.3750f, make_vec3f (0.9961f, 0.7686f, 0.3098f));
    add_point (0.5000f, make_vec3f (0.9961f, 0.6000f, 0.1608f));
    add_point (0.6250f, make_vec3f (0.9255f, 0.4392f, 0.0784f));
    add_point (0.7500f, make_vec3f (0.8000f, 0.2980f, 0.0078f));
    add_point (0.8750f, make_vec3f (0.6000f, 0.2039f, 0.0157f));
    add_point (1.0000f, make_vec3f (0.4000f, 0.1451f, 0.0235f));
  }
  else if (name == "YlOrRd")
  {
    add_point (0.0000f, make_vec3f (1.0000f, 1.0000f, 0.8000f));
    add_point (0.1250f, make_vec3f (1.0000f, 0.9294f, 0.6275f));
    add_point (0.2500f, make_vec3f (0.9961f, 0.8510f, 0.4627f));
    add_point (0.3750f, make_vec3f (0.9961f, 0.6980f, 0.2980f));
    add_point (0.5000f, make_vec3f (0.9922f, 0.5529f, 0.2353f));
    add_point (0.6250f, make_vec3f (0.9882f, 0.3059f, 0.1647f));
    add_point (0.7500f, make_vec3f (0.8902f, 0.1020f, 0.1098f));
    add_point (0.8750f, make_vec3f (0.7412f, 0.0000f, 0.1490f));
    add_point (1.0000f, make_vec3f (0.5020f, 0.0000f, 0.1490f));
  }
  else if (name == "HotAndCold")
  {
    add_point (0.00f, make_vec3f (0.0f, 1.0f, 1.0f));
    add_point (0.45f, make_vec3f (0.0f, 0.0f, 1.0f));
    add_point (0.50f, make_vec3f (0.0f, 0.0f, 0.5f));
    add_point (0.55f, make_vec3f (1.0f, 0.0f, 0.0f));
    add_point (1.00f, make_vec3f (1.0f, 1.0f, 0.0f));
  }
  else if (name == "ColdAndHot")
  {
    add_point (0.00f, make_vec3f (1.0f, 1.0f, 0.0f));
    add_point (0.45f, make_vec3f (1.0f, 0.0f, 0.0f));
    add_point (0.50f, make_vec3f (0.0f, 0.0f, 0.5f));
    add_point (0.55f, make_vec3f (0.0f, 0.0f, 1.0f));
    add_point (1.00f, make_vec3f (0.0f, 1.0f, 1.0f));
  }
  else if (name == "rambo")
  {
    add_point (0.000f, make_vec3f (1.000000f, 1.00000f, 1.000000f));
    add_point (0.143f, make_vec3f (0.000000f, 0.00000f, 0.356862f));
    add_point (0.285f, make_vec3f (0.000000f, 1.00000f, 1.000000f));
    add_point (0.427f, make_vec3f (0.000000f, 0.49803f, 0.000000f));
    add_point (0.571f, make_vec3f (1.000000f, 1.00000f, 0.000000f));
    add_point (0.714f, make_vec3f (1.000000f, 0.37647f, 0.000000f));
    add_point (0.857f, make_vec3f (0.878430f, 0.29803f, 0.298030f));
    add_point (1.000f, make_vec3f (0.419607f, 0.00000f, 0.000000f));

  }
  else if(additional_color_tables.find(name) != additional_color_tables.end())
  {
    std::vector<float> &table = additional_color_tables[name];
    const size_t num_points = table.size() / 4;
    for(size_t i = 0; i < num_points; ++i)
    {
      const size_t offset = i * 4;
      add_point(table[offset],
                make_vec3f(table[offset + 1],
                           table[offset + 2],
                           table[offset + 3]));
    }
  }
  else
  {
    std::cout << "Unknown Color Table: '" << name << "' defaulting" << std::endl;
    add_point (0.0000f, make_vec3f (1.0000f, 1.0000f, 0.8000f));
    add_point (0.1250f, make_vec3f (1.0000f, 0.9294f, 0.6275f));
    add_point (0.2500f, make_vec3f (0.9961f, 0.8510f, 0.4627f));
    add_point (0.3750f, make_vec3f (0.9961f, 0.6980f, 0.2980f));
    add_point (0.5000f, make_vec3f (0.9922f, 0.5529f, 0.2353f));
    add_point (0.6250f, make_vec3f (0.9882f, 0.3059f, 0.1647f));
    add_point (0.7500f, make_vec3f (0.8902f, 0.1020f, 0.1098f));
    add_point (0.8750f, make_vec3f (0.7412f, 0.0000f, 0.1490f));
    add_point (1.0000f, make_vec3f (0.5020f, 0.0000f, 0.1490f));
  }
  this->m_internals->m_name = name;
}

ColorTable::ColorTable (const Vec<float32, 4> &color)
: m_internals (new detail::ColorTableInternals)
{
  this->m_internals->m_name = "";
  this->m_internals->m_smooth = false;

  add_point (0, color);
  add_point (1, color);
}

std::vector<std::string> ColorTable::get_presets ()
{
  std::vector<std::string> res = {
    // built into this file
    "grey",
    "blue",
    "orange",
    "cool2warm",
    "temperature",
    "rainbow",
    "levels",
    "dense",
    "thermal",
    "IsoL",
    "CubicL",
    "CubicYF",
    "LinearL",
    "LinLhot",
    "PuRd",
    "Accent",
    "Blues",
    "BrBG",
    "BuGn",
    "BuPu",
    "Dark2",
    "GnBu",
    "Greens",
    "Greys",
    "Oranges",
    "OrRd",
    "Paired",
    "Pastel1",
    "Pastel2",
    "PiYG",
    "PRGn",
    "PuBu",
    "PuBuGn",
    "PuOr",
    "PuRd",
    "Purples",
    "RdBu",
    "RdGy",
    "RdPu",
    "RdYlBu",
    "RdYlGn",
    "Reds",
    "Set1",
    "Set2",
    "Set3",
    "Spectral",
    "YlGnBu",
    "YlGn",
    "YlOrBr",
    "YlOrRd",
    "HotAndCold",
    "ColdAndHot",
    "rambo",
    // additional color tables
    "3w_gby",
    "4Wmed8",
    "gr-insert_40-50",
    "4-wave-yellow-green-teal-gray",
    "5-wave-yellow-to-blue",
    "5wave-yellow-brown-blue",
    "5w_BRgpb",
    "4w_ROTB",
    "3-wave-muted",
    "5-wave-yellow-green",
    "gr-insert_10-20",
    "gr-insert_0-10",
    "5-wave-orange-to-green",
    "3w_bgYr",
    "3w_bGrBr",
    "3-wave-yellow-grey-blue",
    "gr-insert_30-40",
    "gr-insert_60-70",
    "4-wave-orange-green-blue-gray",
    "gr-insert_50-60",
    "4wave-bgyGr",
    "4w_bgTR",
    "gr-insert_80-100",
    "gr-insert_90-100",
    "gr-insert_20-30",
    "gr-insert_80-90",
    "4w_bgby"
  };

  return res;
}

} // namespace dray
