// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/exports.hpp>
#include <dray/array_internals_base.hpp>
#include <dray/array_registry.hpp>

#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/util/MemoryResourceTraits.hpp>

#include <algorithm>
#include <iostream>

namespace dray
{

std::list<ArrayInternalsBase *> ArrayRegistry::m_arrays;

int ArrayRegistry::m_device_allocator_id = -1;
int ArrayRegistry::m_host_allocator_id = -1;
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

bool ArrayRegistry::device_allocator_id(int id)
{

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

  if(!valid_id)
  {
    return false;
  }

  auto resource = allocator.getAllocationStrategy()->getTraits().resource;

  bool can_use = false;
  bool need_device = false;

#ifdef DRAY_CUDA_ENABLED
  need_device = true;
#elif defined(DRAY_HIP_ENABLED)
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
  // of all currently allocated deviec resources.
  // Data will be preserved by a synch to host
  if(id != m_device_allocator_id)
  {
    release_device_res();
    m_device_allocator_id = id;
  }
  m_external_device_allocator = true;
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
