// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "ascent_array_internals_base.hpp"
#include "ascent_array_registry.hpp"

#include <algorithm>
#include <iostream>

namespace ascent
{

namespace runtime
{

std::list<ArrayInternalsBase *> ArrayRegistry::m_arrays;
size_t ArrayRegistry::m_high_water_mark = 0;
size_t ArrayRegistry::m_device_bytes = 0;
size_t ArrayRegistry::m_host_bytes = 0;

void ArrayRegistry::add_array (ArrayInternalsBase *array)
{
  m_arrays.push_front (array);
}

void ArrayRegistry::remove_array (ArrayInternalsBase *array)
{
  auto it =
  std::find_if (m_arrays.begin (), m_arrays.end (),
                [=] (ArrayInternalsBase *other) { return other == array; });
  if (it == m_arrays.end ())
  {
    std::cerr << "Registry: cannot remove array " << array << "\n";
  }

  m_arrays.remove (array);
}

size_t ArrayRegistry::device_usage ()
{
 return  m_device_bytes;
}

size_t ArrayRegistry::high_water_mark()
{
  return m_high_water_mark;
}

size_t ArrayRegistry::host_usage ()
{
  return m_host_bytes;
}

void ArrayRegistry::release_device_res ()
{
  for (auto b = m_arrays.begin (); b != m_arrays.end (); ++b)
  {
    (*b)->release_device_ptr ();
  }
}

void ArrayRegistry::add_device_bytes(size_t bytes)
{
  m_device_bytes += bytes;
  m_high_water_mark = std::max(m_high_water_mark, m_device_bytes);
}

void ArrayRegistry::remove_device_bytes(size_t bytes)
{
  m_device_bytes -= bytes;
}

void ArrayRegistry::add_host_bytes(size_t bytes)
{
  m_host_bytes += bytes;
}

void ArrayRegistry::remove_host_bytes(size_t bytes)
{
  m_host_bytes += bytes;
}

} // namespace runtime
} // namespace ascent
