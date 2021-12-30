// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "ascent_array.hpp"
#include "ascent_array_internals.hpp"

namespace ascent
{

namespace runtime
{

template <typename T>
Array<T>::Array () : m_internals (new ArrayInternals<T> ()){};

template <typename T>
Array<T>::Array (T *data, const size_t size)
: m_internals (new ArrayInternals<T> (data, size)){};

template <typename T> void Array<T>::set (T *data, const size_t size)
{
  m_internals->set (data, size);
};

template <typename T> void Array<T>::copy (const T *data, const size_t size)
{
  m_internals->copy (data, size);
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

template <typename T> T Array<T>::get_value (const size_t i) const
{
  return m_internals->get_value (i);
}

// Type Explicit instatiations
template class Array<unsigned char>;
template class Array<int>;
template class Array<long long int>;
template class Array<float>;
template class Array<double>;

} // namespace runtime
} // namespace ascent
