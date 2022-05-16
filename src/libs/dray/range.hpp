// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RANGE_HPP
#define DRAY_RANGE_HPP

#include <dray/exports.hpp>
#include <dray/math.hpp>

#include <assert.h>
#include <iostream>

namespace dray
{



class Range
{
  protected:
  Float m_min = infinity<Float>();
  Float m_max = neg_infinity<Float>();

  public:

  DRAY_EXEC void reset ()
  {
    m_min = infinity<Float>();
    m_max = neg_infinity<Float>();
  }

  DRAY_EXEC
  Float min() const
  {
    return m_min;
  }

  DRAY_EXEC
  Float max() const
  {
    return m_max;
  }

  DRAY_EXEC
  void set_min(const Float new_min)
  {
    m_min = new_min;
  }

  DRAY_EXEC
  void set_max(const Float new_max)
  {
    m_max = new_max;
  }

  DRAY_EXEC
  void set_range(const Float new_min, const Float new_max)
  {
    m_min = new_min;
    m_max = new_max;
  }

  DRAY_EXEC
  bool is_empty () const
  {
    return m_min > m_max;
  }

  template <typename T> DRAY_EXEC void include (const T &val)
  {
    m_min = fmin (m_min, Float (val));
    m_max = fmax (m_max, Float (val));
  }

  DRAY_EXEC
  void include (const Range &other)
  {
    if (!other.is_empty ())
    {
      include (other.min ());
      include (other.max ());
    }
  }

  DRAY_EXEC
  bool is_contained_in (const Range &other)
  {
    return is_empty () || (other.m_min <= m_min && m_max <= other.m_max);
  }

  DRAY_EXEC
  bool contains (Float value) const
  {
    return (m_min <= value && value <= m_max);
  }

  DRAY_EXEC
  bool contains (const Range &other)
  {
    return other.is_empty () || (m_min <= other.m_min && other.m_max <= m_max);
  }

  DRAY_EXEC
  Range identity () const
  {
    return Range ();
  }

  DRAY_EXEC
  static Range mult_identity ()
  {
    Range ret;
    ret.m_min = neg_infinity<Float> ();
    ret.m_max = infinity<Float>();
    return ret;
  }

  DRAY_EXEC
  static Range ref_universe ()
  {
    Range ret;
    ret.m_min = 0.f;
    ret.m_max = 1.0;
    return ret;
  }

  DRAY_EXEC
  Float center () const
  {
    if (is_empty ())
    {
      return nan32 ();
    }
    else
      return 0.5f * (m_min + m_max);
  }

  DRAY_EXEC
  void split (Float alpha, Range &left, Range &right) const
  {
    left.m_min = m_min;
    right.m_max = m_max;

    left.m_max = right.m_min = m_min * (1.0 - alpha) + m_max * alpha;
  }

  DRAY_EXEC
  Float length () const
  {
    if (is_empty ())
    {
      // should this just return 0?
      return nan32 ();
    }
    else
      return m_max - m_min;
  }

  DRAY_EXEC
  void scale (Float scale)
  {
    if (is_empty ())
    {
      return;
    }

    Float c = center ();
    Float delta = scale * 0.5f * length ();
    include (c - delta);
    include (c + delta);
  }


  DRAY_EXEC
  Range operator+ (const Range &other) const
  {
    Range res;
    res.include (*this);
    res.include (other);
    return res;
  }

  DRAY_EXEC
  Range intersect (const Range &other) const
  {
    Range res;
    res.m_min = ::max (m_min, other.m_min);
    res.m_max = ::min (m_max, other.m_max);

    return res;
  }

  DRAY_EXEC
  Range split ()
  {
    assert (!is_empty ());
    Range other_half (*this);
    const Float mid = center ();
    m_min = mid;
    other_half.m_max = mid;
    return other_half;
  }


  friend std::ostream &operator<<(std::ostream &os, const Range &range);
};

inline std::ostream &operator<< (std::ostream &os, const Range &range)
{
  os << "[";
  os << range.min () << ", ";
  os << range.max ();
  os << "]";
  return os;
}

} // namespace dray
#endif
