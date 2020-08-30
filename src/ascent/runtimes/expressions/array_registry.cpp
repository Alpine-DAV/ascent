// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "array_internals_base.hpp"
#include "array_registry.hpp"

#include <algorithm>
#include <iostream>

namespace ascent
{

namespace runtime
{

std::list<ArrayInternalsBase *> ArrayRegistry::m_arrays;
size_t ArrayRegistry::m_high_water_mark = 0;

void ArrayRegistry::add_array (ArrayInternalsBase *array)
{
  m_arrays.push_front (array);
  size_t current_usage = device_usage();
  m_high_water_mark = std::max(m_high_water_mark, current_usage);
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

  size_t current_usage = device_usage();
  m_high_water_mark = std::max(m_high_water_mark, current_usage);
  m_arrays.remove (array);
}

size_t ArrayRegistry::device_usage ()
{
  size_t tot = 0;
  for (auto b = m_arrays.begin (); b != m_arrays.end (); ++b)
  {
    tot += (*b)->device_alloc_size ();
  }
  return tot;
}

size_t ArrayRegistry::high_water_mark()
{
  return m_high_water_mark;
}

size_t ArrayRegistry::host_usage ()
{
  size_t tot = 0;
  for (auto b = m_arrays.begin (); b != m_arrays.end (); ++b)
  {
    tot += (*b)->host_alloc_size ();
  }
  return tot;
}

void ArrayRegistry::release_device_res ()
{
  for (auto b = m_arrays.begin (); b != m_arrays.end (); ++b)
  {
    (*b)->release_device_ptr ();
  }
}

} // namespace runtime
} // namespace ascent
