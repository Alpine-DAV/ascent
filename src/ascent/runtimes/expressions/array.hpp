// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef ASCENT_ARRAY_HPP
#define ASCENT_ARRAY_HPP

#include <memory>

namespace ascent
{

namespace runtime
{

// forward declaration of internals
template <typename t> class ArrayInternals;

template <typename T> class Array
{
  public:
  Array ();
  Array (const T *data, const int size);
  ~Array ();

  size_t size () const;
  void resize (const size_t size);
  void set (const T *data, const int size);
  T *get_host_ptr ();
  T *get_device_ptr ();
  const T *get_host_ptr_const () const;
  const T *get_device_ptr_const () const;
  void summary ();
  void operator= (const Array<T> &other);
  // gets a single value and does not synch data between
  // host and device
  T get_value (const int i) const;
  Array<T> copy ();

  protected:
  std::shared_ptr<ArrayInternals<T>> m_internals;
};

} // namespace runtime
} // namespace ascent
#endif
