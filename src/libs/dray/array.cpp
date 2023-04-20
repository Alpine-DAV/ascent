// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/array.hpp>
#include <dray/array_internals.hpp>

namespace dray
{

template <typename T>
Array<T>::Array () : m_internals (new ArrayInternals<T> ()){};

template <typename T>
Array<T>::Array (const T *data, const int32 size)
: m_internals (new ArrayInternals<T> (data, size)){};

template <typename T> void Array<T>::set (const T *data, const int32 size)
{
  m_internals->set (data, size);
};

template <typename T> Array<T>::~Array ()
{
}

template <typename T> void Array<T>::operator= (const Array<T> &other)
{
  m_internals = other.m_internals;
}

template <typename T> size_t Array<T>::size () const
{
  return m_internals->size ();
}

template <typename T> void Array<T>::resize (const size_t size)
{
  m_internals->resize (size);
}

template <typename T> T *Array<T>::get_host_ptr ()
{
  return m_internals->get_host_ptr ();
}

template <typename T> T *Array<T>::get_device_ptr ()
{
  return m_internals->get_device_ptr ();
}

template <typename T> const T *Array<T>::get_host_ptr_const () const
{
  return m_internals->get_host_ptr_const ();
}

template <typename T> const T *Array<T>::get_device_ptr_const () const
{
  return m_internals->get_device_ptr_const ();
}

template <typename T> void Array<T>::summary ()
{
  m_internals->summary ();
}

template <typename T> T Array<T>::get_value (const int32 i) const
{
  return m_internals->get_value (i);
}

// Type Explicit instatiations
template class Array<int8>;
template class Array<uint8>;
template class Array<int32>;
template class Array<uint32>;
template class Array<int64>;
template class Array<uint64>;
template class Array<float32>;
template class Array<float64>;

} // namespace dray

// Class Explicit instatiations
#include <dray/aabb.hpp>
template class dray::Array<dray::AABB<1>>;
template class dray::Array<dray::AABB<2>>;
template class dray::Array<dray::AABB<3>>;
template class dray::Array<dray::AABB<4>>;

#include <dray/vec.hpp>
template class dray::Array<dray::Vec<dray::uint32, 2>>;

template class dray::Array<dray::Vec<dray::float32, 1>>;
template class dray::Array<dray::Vec<dray::float64, 1>>;

template class dray::Array<dray::Vec<dray::float32, 2>>;
template class dray::Array<dray::Vec<dray::float64, 2>>;

template class dray::Array<dray::Vec<dray::float32, 3>>;
template class dray::Array<dray::Vec<dray::float64, 3>>;

template class dray::Array<dray::Vec<dray::float32, 4>>;
template class dray::Array<dray::Vec<dray::float64, 4>>;

template class dray::Array<dray::Vec<dray::int32, 2>>;
template class dray::Array<dray::Vec<dray::int32, 3>>;
template class dray::Array<dray::Vec<dray::int32, 4>>;

#include <dray/matrix.hpp>
template class dray::Array<dray::Matrix<dray::float32, 2, 2>>;
template class dray::Array<dray::Matrix<dray::float64, 2, 2>>;

template class dray::Array<dray::Matrix<dray::float32, 3, 3>>;
template class dray::Array<dray::Matrix<dray::float64, 3, 3>>;

template class dray::Array<dray::Matrix<dray::float32, 4, 4>>;
template class dray::Array<dray::Matrix<dray::float64, 4, 4>>;

template class dray::Array<dray::Matrix<dray::float32, 4, 3>>;
template class dray::Array<dray::Matrix<dray::float64, 4, 3>>;

template class dray::Array<dray::Matrix<dray::float32, 4, 1>>;
template class dray::Array<dray::Matrix<dray::float64, 4, 1>>;

template class dray::Array<dray::Matrix<dray::float32, 3, 1>>;
template class dray::Array<dray::Matrix<dray::float64, 3, 1>>;

template class dray::Array<dray::Matrix<dray::float32, 1, 3>>;
template class dray::Array<dray::Matrix<dray::float64, 1, 3>>;

#include <dray/data_model/subref.hpp>
template class dray::Array<dray::SubRef<2, dray::ElemType::Simplex>>;
template class dray::Array<dray::SubRef<2, dray::ElemType::Tensor>>;
template class dray::Array<dray::SubRef<3, dray::ElemType::Simplex>>;
template class dray::Array<dray::SubRef<3, dray::ElemType::Tensor>>;

#include <dray/data_model/iso_ops.hpp>
template class dray::Array<dray::eops::IsocutInfo>;

#include <dray/ref_point.hpp>
template class dray::Array<dray::RefPoint<3>>;
template class dray::Array<dray::RefPoint<2>>;

#include <dray/ray.hpp>
template class dray::Array<dray::Ray>;

#include <dray/ray_hit.hpp>
template class dray::Array<dray::RayHit>;

#include <dray/location.hpp>
template class dray::Array<dray::Location>;

#include <dray/intersection_context.hpp>
template class dray::Array<dray::IntersectionContext>;

#include <dray/rendering/fragment.hpp>
template class dray::Array<dray::Fragment>;

#include <dray/utils/appstats.hpp>
template class dray::Array<dray::stats::Stats>;

#include <dray/rendering/point_light.hpp>
template class dray::Array<dray::PointLight>;

#include <dray/rendering/volume_partial.hpp>
template class dray::Array<dray::VolumePartial>;

#include <dray/rendering/material.hpp>
template class dray::Array<dray::Material>;
