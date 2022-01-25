//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ascent_array_internals_base.hpp"
#include "ascent_array_registry.hpp"

#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>


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

int ArrayRegistry::m_device_allocator_id = -1;
int ArrayRegistry::m_host_allocator_id = -1;
bool ArrayRegistry::m_external_device_allocator = false;

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

void ArrayRegistry::reset_high_water_mark()
{
  m_high_water_mark = 0;
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
  // release if we own the allocator
  if(m_device_allocator_id == -1 && !m_external_device_allocator)
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator(m_device_allocator_id);
    allocator.release();
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
  m_host_bytes -= bytes;
}

int ArrayRegistry::device_allocator_id()
{
  if(m_device_allocator_id == -1)
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator("DEVICE");
    // we can use the umpire profiling to find a good default size
    auto pooled_allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
                            "GPU_POOL",
                            allocator,
                            1ul * // 1GB default size
                            1024ul * 1024ul * 1024ul + 1);
    m_device_allocator_id = pooled_allocator.getId();
  }
  return m_device_allocator_id;
}

void ArrayRegistry::device_allocator_id(int id)
{
  // if this is not the same, we have to get rid
  // of all currently allocated deviec resources.
  // Data will be preserved by a synch to host
  if(id != m_device_allocator_id)
  {
    release_device_res();
    m_device_allocator_id = id;
  }
  m_external_device_allocator = true;
}

int ArrayRegistry::host_allocator_id()
{
  if(m_host_allocator_id == -1)
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator("HOST");
    // we can use the umpire profiling to find a good default size
    // Matt: something weird happened in the test suite where running multiple
    // tests resulted is us trying to create this allocator twice, so I added
    // the check to see if it already exists, which fixed the issue. This
    // is most likely related to static stuff
    if(!rm.isAllocator("HOST_POOL"))
    {
      auto pooled_allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
                              "HOST_POOL",
                              allocator,
                              1ul * // 1GB default size
                              1024ul * 1024ul * 1024ul + 1);
      m_host_allocator_id = pooled_allocator.getId();
    }
    else
    {
      auto pooled_allocator = rm.getAllocator("HOST_POOL");
      m_host_allocator_id = pooled_allocator.getId();
    }
  }
  return m_host_allocator_id;
}

} // namespace runtime
} // namespace ascent
