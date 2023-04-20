// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Note: this file was derived from
// https://gitlab.kitware.com/vtk/vtk-m/blob/master/vtkm/Types.h

#ifndef DRAY_VEC_HPP
#define DRAY_VEC_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/math.hpp>
#include <dray/types.hpp>

#include <assert.h>
#include <iostream>

namespace dray
{

template <typename T, int32 S> class Vec
{

  public:
  T m_data[S];

  //
  //  No contructors so this is a POD type
  //

  DRAY_EXEC bool operator== (const Vec<T, S> &other) const
  {
    bool e = true;
    for (int i = 0; i < S; ++i)
    {
      if (m_data[i] != other[i]) e = false;
    }
    return e;
  }

  template <typename TT, int32 SS>
  friend std::ostream &operator<< (std::ostream &os, const Vec<TT, SS> &vec);

  template <typename TT> DRAY_EXEC void operator= (const Vec<TT, S> &other)
  {
    for (int i = 0; i < S; ++i)
    {
      m_data[i] = other.m_data[i];
    }
  }

  template <typename TT> DRAY_EXEC void operator= (const TT &scalar)
  {
    for (int i = 0; i < S; ++i)
    {
      m_data[i] = scalar;
    }
  }

  DRAY_EXEC void operator= (const T &single_val)
  {
    for (int i = 0; i < S; ++i)
    {
      m_data[i] = single_val;
    }
  }

  // DRAY_EXEC const T operator[](const int32 &i) const
  //{
  //  assert(i > -1 && i < S);
  //  return m_data[i];
  //}

  DRAY_EXEC const T &operator[] (const int32 &i) const
  {
    assert (i > -1 && i < S);
    return m_data[i];
  }

  DRAY_EXEC T &operator[] (const int32 &i)
  {
    assert (i > -1 && i < S);
    return m_data[i];
  }

  // scalar mult /  div
  template <typename TT> DRAY_EXEC Vec<T, S> operator* (const TT &s) const
  {
    Vec<T, S> res;

    for (int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] * s;
    }

    return res;
  }

  DRAY_EXEC Vec<T, S> operator/ (const T &s) const
  {
    Vec<T, S> res;

    for (int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] / s;
    }

    return res;
  }

  DRAY_EXEC void operator*= (const T &s)
  {
    for (int i = 0; i < S; ++i)
    {
      m_data[i] *= s;
    }
  }

  DRAY_EXEC void operator/= (const T &s)
  {
    for (int i = 0; i < S; ++i)
    {
      m_data[i] /= s;
    }
  }

  // vector add / sub

  DRAY_EXEC Vec<T, S> operator+ (const Vec<T, S> &other) const
  {
    Vec<T, S> res;

    for (int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] + other[i];
    }

    return res;
  }

  DRAY_EXEC Vec<T, S> operator- (const Vec<T, S> &other) const
  {
    Vec<T, S> res;

    for (int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] - other[i];
    }

    return res;
  }

  template<typename TT>
  DRAY_EXEC Vec<T, S> operator- (const Vec<TT, S> &other) const
  {
    Vec<T, S> res;

    for (int i = 0; i < S; ++i)
    {
      res[i] = m_data[i] - static_cast<T>(other[i]);
    }

    return res;
  }

  DRAY_EXEC void operator+= (const Vec<T, S> &other)
  {

    for (int i = 0; i < S; ++i)
    {
      m_data[i] += other[i];
    }
  }

  DRAY_EXEC void operator-= (const Vec<T, S> &other)
  {

    for (int i = 0; i < S; ++i)
    {
      m_data[i] -= other[i];
    }
  }

  DRAY_EXEC Vec<T, S> operator- (void) const
  {
    Vec<T, S> res;

    for (int i = 0; i < S; ++i)
    {
      res[i] = -m_data[i];
    }

    return res;
  }


  DRAY_EXEC T magnitude2 () const
  {
    T sum = T (0);

    for (int i = 0; i < S; ++i)
    {
      sum += m_data[i] * m_data[i];
    }

    return sum;
  }


  DRAY_EXEC T magnitude () const
  {
    return sqrtf (magnitude2 ());
  }


  DRAY_EXEC void normalize ()
  {
    T mag = magnitude ();
    *this /= mag;
  }

  DRAY_EXEC Vec normalized() const
  {
    const T mag = magnitude();
    return *this / mag;
  }

  DRAY_EXEC T Normlinf () const // Used for convergence tests.
  {
    // Max{ abs(x_i) } over all components.
    T max_c = fmaxf (-m_data[0], m_data[0]);
    for (int ii = 1; ii < S; ++ii)
    {
      max_c = fmaxf (max_c, fmaxf (-m_data[ii], m_data[ii]));
    }
    return max_c;
  }

  DRAY_EXEC void swap (Vec &that)
  {
    T tmp;
    for (int ii = 0; ii < S; ii++)
    {
      tmp = this->m_data[ii];
      this->m_data[ii] = that.m_data[ii];
      that.m_data[ii] = tmp;
    }
  }

  DRAY_EXEC static constexpr int32 size ()
  {
    return S;
  }

  DRAY_EXEC static Vec<T, S> zero ()
  {
    Vec<T, S> res;

    for (int i = 0; i < S; ++i)
    {
      res[i] = 0;
    }

    return res;
  }
};


// constexpr -
template <typename T>
DRAY_EXEC constexpr Vec<T, 2> minus(const Vec<T, 2> &a, const Vec<T, 2> &b)
{
  return {{T(a[0]-b[0]), T(a[1]-b[1])}};
}
template <typename T>
DRAY_EXEC constexpr Vec<T, 3> minus(const Vec<T, 3> &a, const Vec<T, 3> &b)
{
  return {{T(a[0]-b[0]), T(a[1]-b[1]), T(a[2]-b[2])}};
}

// constexpr +
template <typename T>
DRAY_EXEC constexpr Vec<T, 2> plus(const Vec<T, 2> &a, const Vec<T, 2> &b)
{
  return {{T(a[0]+b[0]), T(a[1]+b[1])}};
}
template <typename T>
DRAY_EXEC constexpr Vec<T, 3> plus(const Vec<T, 3> &a, const Vec<T, 3> &b)
{
  return {{T(a[0]+b[0]), T(a[1]+b[1]), T(a[2]+b[2])}};
}


// vector utility functions
// scalar mult
template <typename T, int32 S>
DRAY_EXEC Vec<T, S> operator* (const T &s, const Vec<T, S> &vec)
{
  return vec * s;
}

template <typename T, int32 S>
DRAY_EXEC Vec<T, S> hadamard(const Vec<T, S> &a, const Vec<T, S> &b)
{
  Vec<T, S> res;
  for (int i = 0; i < S; ++i)
    res[i] = a[i] * b[i];
  return res;
}

template <typename T, int32 S>
DRAY_EXEC T dot (const Vec<T, S> &a, const Vec<T, S> &b)
{
  T res = T (0);

  for (int i = 0; i < S; ++i)
  {
    res += a[i] * b[i];
  }

  return res;
}

template <typename T, int32 S_out, int32 S_in>
DRAY_EXEC Vec<T, S_in> dot (const Vec<Vec<T, S_in>, S_out> &a, const Vec<T, S_out> &b)
{
  Vec<T, S_in> res;
  res = 0;
  for (int i = 0; i < S_out; ++i)
  {
    res += a[i] * b[i];
  }
  return res;
}

template <typename T>
DRAY_EXEC Vec<T, 3> cross (const Vec<T, 3> &a, const Vec<T, 3> &b)
{
  Vec<T, 3> res;
  res[0] = a[1] * b[2] - a[2] * b[1];
  res[1] = a[2] * b[0] - a[0] * b[2];
  res[2] = a[0] * b[1] - a[1] * b[0];
  return res;
}

template <typename TT, int32 SS>
std::ostream &operator<< (std::ostream &os, const Vec<TT, SS> &vec)
{
  os << "[";
  for (int i = 0; i < SS; ++i)
  {
    os << vec[i];
    if (i != SS - 1) os << ", ";
  }
  os << "]";
  return os;
}

template <typename TS, typename TD>
DRAY_EXEC void type_convert(const Vec<TS, 2> &in, Vec<TD, 2> &out)
{
    out[0] = static_cast<TD>(in[0]);
    out[1] = static_cast<TD>(in[1]);
}

template <typename TS, typename TD>
DRAY_EXEC void type_convert(const Vec<TS, 3> &in, Vec<TD, 3> &out)
{
    out[0] = static_cast<TD>(in[0]);
    out[1] = static_cast<TD>(in[1]);
    out[2] = static_cast<TD>(in[2]);
}

template <typename TS, typename TD>
DRAY_EXEC void type_convert(const Vec<TS, 4> &in, Vec<TD, 4> &out)
{
    out[0] = static_cast<TD>(in[0]);
    out[1] = static_cast<TD>(in[1]);
    out[2] = static_cast<TD>(in[2]);
    out[3] = static_cast<TD>(in[3]);
}


// typedefs
typedef Vec<int32, 2> Vec2i;
typedef Vec<int64, 2> Vec2li;
typedef Vec<float32, 2> Vec2f;
typedef Vec<float64, 2> Vec2d;

typedef Vec<int32, 3> Vec3i;
typedef Vec<int64, 3> Vec3li;
typedef Vec<float32, 3> Vec3f;
typedef Vec<float64, 3> Vec3d;

typedef Vec<int32, 4> Vec4i;
typedef Vec<int64, 4> Vec4li;
typedef Vec<float32, 4> Vec4f;
typedef Vec<float64, 4> Vec4d;

DRAY_EXEC
Vec2i make_vec2i (const int32 &a, const int32 &b)
{
  Vec2i res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec2li make_vec2li (const int64 &a, const int64 &b)
{
  Vec2li res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec2f make_vec2f (const float32 &a, const float32 &b)
{
  Vec2f res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec2d make_vec2d (const float64 &a, const float64 &b)
{
  Vec2d res;
  res[0] = a;
  res[1] = b;
  return res;
}

DRAY_EXEC
Vec3i make_vec3i (const int32 &a, const int32 &b, const int32 &c)
{
  Vec3i res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec3li make_vec3li (const int64 &a, const int64 &b, const int64 &c)
{
  Vec3li res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec3f make_vec3f (const float32 &a, const float32 &b, const float32 &c)
{
  Vec3f res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec3d make_vec3d (const float64 &a, const float64 &b, const float64 &c)
{
  Vec3d res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  return res;
}

DRAY_EXEC
Vec4i make_vec4i (const int32 &a, const int32 &b, const int32 &c, const int32 &d)
{
  Vec4i res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}

DRAY_EXEC
Vec4li make_vec4li (const int64 &a, const int64 &b, const int64 &c, const int64 &d)
{
  Vec4li res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}

DRAY_EXEC
Vec4f make_vec4f (const float32 &a, const float32 &b, const float32 &c, const float32 &d)
{
  Vec4f res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}

DRAY_EXEC
Vec4d make_vec4d (const float64 &a, const float64 &b, const float64 &c, const float64 &d)
{
  Vec4d res;
  res[0] = a;
  res[1] = b;
  res[2] = c;
  res[3] = d;
  return res;
}


//
// MultiVecRange
//
template <typename ComponentT, uint32 RangeSize> struct MultiVecRange
{
  ComponentT &m_ref;

  DRAY_EXEC MultiVecRange (ComponentT &ref) : m_ref (ref)
  {
  }
  DRAY_EXEC ComponentT *begin () const
  {
    return &m_ref;
  }
  DRAY_EXEC ComponentT *end () const
  {
    return &m_ref + RangeSize;
  }
};

template <typename ComponentT, uint32 RangeSize> struct ConstMultiVecRange
{
  const ComponentT &m_ref;

  DRAY_EXEC ConstMultiVecRange (const ComponentT &ref) : m_ref (ref)
  {
  }
  DRAY_EXEC const ComponentT *begin () const
  {
    return &m_ref;
  }
  DRAY_EXEC const ComponentT *end () const
  {
    return &m_ref + RangeSize;
  }
};


//
// FirstComponent
//
template <typename MultiArrayT, uint32 Depth> struct FirstComponent
{
  using component_t =
  typename FirstComponent<decltype (std::declval<MultiArrayT> ()[0]), Depth - 1>::component_t;

  DRAY_EXEC static component_t &of (MultiArrayT &multi_array)
  {
    return FirstComponent<decltype ((multi_array[0])), Depth - 1>::of (multi_array[0]);
  }

  /// DRAY_EXEC static const component_t & of(const MultiArrayT &multi_array)
  /// {
  ///   return FirstComponent<decltype((multi_array[0])), Depth-1>::of(multi_array[0]);
  /// }
};

template <typename MultiArrayT> struct FirstComponent<MultiArrayT, 0u>
{
  using component_t = MultiArrayT;

  DRAY_EXEC static MultiArrayT &of (MultiArrayT &multi_array)
  {
    return multi_array;
  }
  /// DRAY_EXEC static const MultiArrayT & of(const MultiArrayT &multi_array) { return multi_array; }
};


//
// MultiVec
//
template <typename T, uint32 RefDim, uint32 PhysDim, uint32 max_p_order>
struct MultiVec : public Vec<MultiVec<T, RefDim - 1, PhysDim, max_p_order>, 1 + max_p_order>
{
  using BaseType = Vec<MultiVec<T, RefDim - 1, PhysDim, max_p_order>, 1 + max_p_order>;

  static constexpr uint32 total_size = intPow (1 + max_p_order, RefDim);

  DRAY_EXEC void operator= (const BaseType &other)
  {
    BaseType::operator= (other);
  }

  DRAY_EXEC Vec<T, PhysDim> &linear_idx (int32 ii)
  {
    return *(&this->operator[] (0).linear_idx (0) + ii);
  }
  DRAY_EXEC const Vec<T, PhysDim> &linear_idx (int32 ii) const
  {
    return *(&this->operator[] (0).linear_idx (0) + ii);
  }

  template <uint32 Depth = RefDim>
  DRAY_EXEC
  MultiVecRange<MultiVec<T, RefDim - Depth, PhysDim, max_p_order>, intPow (1 + max_p_order, Depth)>
  components ()
  {
    return { FirstComponent<MultiVec, Depth>::of (*this) };
  }

  template <uint32 Depth = RefDim>
  DRAY_EXEC
  ConstMultiVecRange<MultiVec<T, RefDim - Depth, PhysDim, max_p_order>, intPow (1 + max_p_order, Depth)>
  components () const
  {
    return { FirstComponent<MultiVec, Depth>::of (*this) };
  }
};

template <typename T, uint32 PhysDim, uint32 max_p_order>
struct MultiVec<T, 0, PhysDim, max_p_order> : public Vec<T, PhysDim>
{
  using BaseType = Vec<T, PhysDim>;

  static constexpr uint32 total_size = 1;

  DRAY_EXEC void operator= (const BaseType &other)
  {
    BaseType::operator= (other);
  }

  DRAY_EXEC Vec<T, PhysDim> &linear_idx (int32 ii)
  {
    return *(this + ii);
  }
  DRAY_EXEC const Vec<T, PhysDim> &linear_idx (int32 ii) const
  {
    return *(this + ii);
  }

  template <uint32 Depth = 0>
  DRAY_EXEC MultiVecRange<MultiVec<T, 0, PhysDim, max_p_order>, 1> components ()
  {
    return { *this };
  }

  template <uint32 Depth = 0>
  DRAY_EXEC ConstMultiVecRange<MultiVec<T, 0, PhysDim, max_p_order>, 1> components () const
  {
    return { *this };
  }
};


} // namespace dray

#endif
