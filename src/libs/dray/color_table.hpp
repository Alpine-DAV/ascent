// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_COLOR_TABLE_HPP
#define DRAY_COLOR_TABLE_HPP

#include <dray/array.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>

#include <memory>
#include <string>
#include <vector>

namespace dray
{
namespace detail
{
struct ColorTableInternals;
}

/// \brief It's a color table!
///
/// This class provides the basic representation of a color table. This class
/// was Ported from EAVL. Originally created by Jeremy Meredith, Dave Pugmire,
/// and Sean Ahern. This class uses separate RGB and alpha control points and
/// can be used as a transfer function.
///
class ColorTable
{
  private:
  std::shared_ptr<detail::ColorTableInternals> m_internals;

  public:
  ColorTable ();

  /// Constructs a \c ColorTable using the name of a pre-defined color set.
  ColorTable (const std::string &name);

  // Make a single color ColorTable.
  ColorTable (const Vec<float32, 4> &color);

  const std::string &get_name () const;

  bool get_smooth () const;

  void set_smooth (bool smooth);

  void sample (int32 num_samples, 
               Array<Vec<float32, 4>> &colors,
               float32 alpha_scale = 1.f) const;

  Vec<float32, 4> map_rgb (float32 scalar) const;

  float32 map_alpha (float32 scalar) const;

  void clear();
  void clear_colors();
  void clear_alphas();

  void reverse ();

  ColorTable correct_opacity (const float32 &factor) const;

  void add_point (float32 position, const Vec<float32, 3> &color);
  void add_point (float32 position, const Vec<float32, 4> &color);

  void add_alpha (float32 position, float32 alpha);

  void update_alpha (const int32 &index, const float32 &position, const float32 &alpha);

  int number_of_alpha_points () const;
  int number_of_points () const;
  // position, value
  Vec<float32, 2> get_alpha (int index) const;
  void remove_alpha (int index);

  void print ();

  static std::vector<std::string> get_presets ();
};
} // namespace dray
#endif
