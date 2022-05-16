// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_AABB_HPP
#define DRAY_AABB_HPP

//#include <dray/exports.hpp>
#include <dray/range.hpp>
#include <dray/vec.hpp>

#include <iostream>

namespace dray
{

template <int32 dim> class AABB;

template <int32 dim>
inline std::ostream &operator<< (std::ostream &os, const AABB<dim> &range);

template <int32 dim = 3> class AABB
{

  public:
  Range m_ranges[dim];

  DRAY_EXEC
  void reset ()
  {
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].reset ();
  }

  // return true if all ranges are empty
  DRAY_EXEC
  bool is_empty() const
  {
    bool empty = true;
    for (int32 d = 0; d < dim; d++)
      empty &= m_ranges[d].is_empty();
    return empty;
  }

  DRAY_EXEC
  void include (const AABB &other)
  {
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].include (other.m_ranges[d]);
  }

  DRAY_EXEC
  void include (const Vec<float32, dim> &point)
  {
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].include (point[d]);
  }

  DRAY_EXEC
  void include (const Vec<float64, dim> &point)
  {
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].include (point[d]);
  }

  DRAY_EXEC
  bool is_contained_in (const AABB &other)
  {
    bool ret = true;
    for (int32 d = 0; d < dim; d++)
      ret &= m_ranges[d].is_contained_in (other.m_ranges[d]);
    return ret;
  }

  DRAY_EXEC
  bool contains (const AABB &other)
  {
    bool ret = true;
    for (int32 d = 0; d < dim; d++)
      ret &= m_ranges[d].contains (other.m_ranges[d]);
    return ret;
  }

  DRAY_EXEC
  bool contains (const Vec<Float,dim> &point)
  {
    bool ret = true;
    for (int32 d = 0; d < dim; d++)
      ret &= m_ranges[d].contains (point[d]);
    return ret;
  }

  DRAY_EXEC
  void expand (const float32 &epsilon)
  {
    assert (epsilon > 0.f);
    for (int32 d = 0; d < dim; d++)
    {
      m_ranges[d].include (m_ranges[d].min () - epsilon);
      m_ranges[d].include (m_ranges[d].max () + epsilon);
    }
  }

  DRAY_EXEC
  void scale (const float32 &scale)
  {
    assert (scale >= 1.f);
    for (int32 d = 0; d < dim; d++)
      m_ranges[d].scale (scale);
  }

  template <typename T = float32> DRAY_EXEC Vec<T, dim> center () const
  {
    Vec<T, dim> center;
    for (int32 d = 0; d < dim; d++)
      center[d] = m_ranges[d].center ();
    return center;
  }

  // Mins of all of the ranges.
  DRAY_EXEC
  Vec<float32, dim> min () const
  {
    Vec<float32, dim> lower_left;
    for (int32 d = 0; d < dim; d++)
      lower_left[d] = m_ranges[d].min ();
    return lower_left;
  }

  // Maxes of all the ranges.
  DRAY_EXEC
  Vec<float32, dim> max () const
  {
    Vec<float32, dim> upper_right;
    for (int32 d = 0; d < dim; d++)
      upper_right[d] = m_ranges[d].max ();
    return upper_right;
  }

  DRAY_EXEC
  int32 max_dim () const
  {
    int32 max_dim = 0;
    float32 max_length = m_ranges[0].length ();
    for (int32 d = 1; d < dim; d++)
    {
      float32 length = m_ranges[d].length ();
      if (length > max_length)
      {
        max_dim = d;
        max_length = length;
      }
    }
    return max_dim;
  }

  DRAY_EXEC
  float32 max_length () const
  {
    float32 max_length = m_ranges[0].length ();
    for (int32 d = 1; d < dim; d++)
    {
      float32 length = m_ranges[d].length ();
      if (length > max_length)
      {
        max_length = length;
      }
    }
    return max_length;
  }

  DRAY_EXEC
  float32 volume() const
  {
    float32 vol = 1.f;
    for (int32 d = 0; d < dim; d++)
      vol *= m_ranges[d].length ();
    return vol;
  }

  DRAY_EXEC
  float32 surface_area() const
  {
    float32 lengths[dim];
    for (int32 d = 0; d < dim; d++)
      lengths[d] = m_ranges[d].length ();
    float32 area = 2.f * lengths[0] * lengths[1];
    if(dim == 3)
    {
      area += 2.f * lengths[1] * lengths[2];
      area += 2.f * lengths[2] * lengths[0];
    }
    return area;
  }

  DRAY_EXEC
  AABB<dim> intersect (const AABB<dim> &other) const
  {
    AABB<dim> res;
    for (int32 d = 0; d < dim; d++)
      res.m_ranges[d] = m_ranges[d].intersect (other.m_ranges[d]);
    return res;
  }

  DRAY_EXEC
  AABB<dim> split (const int split_dim)
  {
    assert (split_dim < dim);
    AABB<dim> other_half (*this);
    other_half.m_ranges[split_dim] = m_ranges[split_dim].split ();
    return other_half;
  }

  DRAY_EXEC
  static AABB universe ()
  {
    AABB universe;
    for (int32 d = 0; d < dim; d++)
      universe.m_ranges[d] = Range::mult_identity ();
    return universe;
  }

  DRAY_EXEC
  static AABB ref_universe ()
  {
    AABB ref_universe;
    for (int32 d = 0; d < dim; d++)
      ref_universe.m_ranges[d] = Range::ref_universe ();
    return ref_universe;
  }

  friend std::ostream &operator<< <dim> (std::ostream &os, const AABB &aabb);
};


template <int32 dim>
inline std::ostream &operator<< (std::ostream &os, const AABB<dim> &aabb)
{
  os << aabb.min () << " - " << aabb.max ();
  return os;
}

} // namespace dray
#endif
