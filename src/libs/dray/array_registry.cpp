// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array_internals_base.hpp>
#include <dray/array_registry.hpp>
#include <dray/error.hpp>

#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/util/MemoryResourceTraits.hpp>

#include <algorithm>
#include <iostream>

namespace dray
{

std::list<ArrayInternalsBase *> ArrayRegistry::m_arrays;

int ArrayRegistry::m_host_allocator_id = -1;
int ArrayRegistry::m_device_allocator_id = -1;
bool ArrayRegistry::m_external_host_allocator = false;
bool ArrayRegistry::m_external_device_allocator = false;

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

//-----------------------------------------------------------------------------
bool
ArrayRegistry::set_device_allocator_id(int id)
{
    if(m_external_device_allocator && m_device_allocator_id != id)
    {
        // We can't change allocators mid stream.
        // This would cause a mismatch between memory allocated with one 'allocator' then that
        // memory being deallocated with another.
        DRAY_ERROR("Changing the host allocator id in the middle of a run is not supported.");
    }

  auto &rm = umpire::ResourceManager::getInstance();
  bool valid_id = true;

  umpire::Allocator allocator;
  try
  {
    allocator = rm.getAllocator (id);
  }
  catch(...)
  {
    valid_id = false;
  }

  if(!valid_id)
  {
    return false;
  }

  auto resource = allocator.getAllocationStrategy()->getTraits().resource;

  bool can_use = false;
  bool need_device = false;


#if defined(DRAY_DEVICE_ENABLED)
  need_device = true;
#endif

  bool is_device = resource == umpire::MemoryResourceTraits::resource_type::device;
  bool is_host = resource == umpire::MemoryResourceTraits::resource_type::host;

  if(is_device && need_device)
  {
    can_use = true;
  }
  else if(is_host && !need_device)
  {
    can_use = true;
  }
  if(!can_use)
  {
    return false;
  }

  // if this is not the same, we have to get rid
  // of all currently allocated device resources.
  // Data will be preserved by a synch to host
  if(id != m_device_allocator_id)
  {
    release_device_res();
    m_device_allocator_id = id;
  }
  m_external_device_allocator = true;
  return true;
}


//-----------------------------------------------------------------------------
bool
ArrayRegistry::set_host_allocator_id(int id)
{
    if(m_external_host_allocator && m_host_allocator_id != id)
    {
        // We can't change allocators mid stream.
        // This would cause a mismatch between memory allocated with one 'allocator' then that
        // memory being deallocated with another.
        DRAY_ERROR("Changing the host allocator id in the middle of a run is not supported.");
    }

    auto &rm = umpire::ResourceManager::getInstance ();
    bool valid_id = true;

    umpire::Allocator allocator;

    try
    {
        allocator = rm.getAllocator (id);
    }
    catch(...)
    {
        valid_id = false;
    }

    auto resource = allocator.getAllocationStrategy()->getTraits().resource;
    // check that this is a host allocator
    bool is_host   = resource == umpire::MemoryResourceTraits::resource_type::host;

    if(!is_host)
    {
        return false;
    }

    m_host_allocator_id = id;
    m_external_host_allocator = true;
    return true;
}

int ArrayRegistry::host_allocator_id()
{
  if(m_host_allocator_id == -1)
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator("HOST");
    // we can use the umpire profiling to find a good default size
    auto pooled_allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
                            "HOST_POOL",
                            allocator,
                            1ul * // 1GB default size
                            1024ul * 1024ul * 1024ul + 1);
    m_host_allocator_id = pooled_allocator.getId();
  }
  return m_host_allocator_id;
}

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

void
ArrayRegistry::summary()
{
    std::cout << "ArrayRegistry Number of Arrays: " << number_of_arrays() << std::endl;
    std::cout  << "Memory Usage:" << std::endl;
    std::cout  << " Host:   " << host_usage() << std::endl;
    std::cout  << " Device: " << device_usage() << std::endl;

    std::cout  << "Umpire pool info:" << std::endl;

    // umpire host pool info
    auto &rm = umpire::ResourceManager::getInstance ();
    const int host_allocator_id = ArrayRegistry::host_allocator_id();
    umpire::Allocator host_allocator = rm.getAllocator(host_allocator_id);
    std::cout  << " Host Current Size:   " << host_allocator.getCurrentSize() << std::endl;
    std::cout  << " Host High Water:     " << host_allocator.getHighWatermark() << std::endl;

#if defined(DRAY_DEVICE_ENABLED)
    // umpire device pool info
    const int dev_allocator_id = ArrayRegistry::device_allocator_id();
    umpire::Allocator dev_allocator = rm.getAllocator(dev_allocator_id);
    std::cout  << " Device Current Size: " << dev_allocator.getCurrentSize() << std::endl;
    std::cout  << " Device High Water:   " << dev_allocator.getHighWatermark() << std::endl;
#else
    std::cout  << "(No Umpire Device Pool [DRAY_DEVICE_ENABLED == FALSE] )" << std::endl;
#endif

}

int ArrayRegistry::number_of_arrays()
{
  return static_cast<int>(m_arrays.size());
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

} // namespace dray
