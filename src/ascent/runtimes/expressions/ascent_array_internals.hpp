// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef ASCENT_ARRAY_INTERNALS
#define ASCENT_ARRAY_INTERNALS

#include <ascent_config.h>
#include "ascent_memory_manager.hpp"
#include "ascent_logging.hpp"

#include <umpire/Umpire.hpp>

#include <iostream>
#include <string.h>

namespace ascent
{

namespace runtime
{

template <typename T> class ArrayInternals
{
  protected:
  T *m_device;
  T *m_host;
  bool m_device_dirty;
  bool m_host_dirty;
  size_t m_size;
  bool m_cuda_enabled;
  bool m_own_host;
  bool m_own_device;

  public:
  ArrayInternals ()
  : m_device (nullptr),
    m_host (nullptr),
    m_device_dirty (true),
    m_host_dirty (true),
    m_size (0),
    m_own_host(true),
    m_own_device(true)
  {
#ifdef ASCENT_USE_ENABLED
    m_cuda_enabled = true;
#else
    m_cuda_enabled = false;
#endif
  }

  void pointer_assignment(T *data)
  {
    bool is_gpu, is_unified;
    is_gpu_ptr(data, is_gpu, is_unified);

    if(is_gpu)
    {
      m_device = data;
      m_own_device = false;
      if(is_unified)
      {
        m_host = data;
        m_own_host = true;
        m_host_dirty = false;
        m_device_dirty = false;
      }
      else
      {
        m_host = nullptr;
        m_own_host = true;
        m_host_dirty = true;
      }
    }
    else
    {
      m_own_device = true;
      m_device = nullptr;
      m_device_dirty = true;

      m_own_host = false;
      m_host_dirty = false;
      m_host = data;
    }
  }

  ArrayInternals (T *data, const size_t size)
  : m_size (size)
  {
#ifdef ASCENT_USE_CUDA
    m_cuda_enabled = true;
#else
    m_cuda_enabled = false;
#endif
    pointer_assignment(data);
  }

  T get_value (const size_t i)
  {
    assert (i < m_size);
    T val = T();
    if (!m_cuda_enabled)
    {
      if (m_host == nullptr)
      {
        // ask for garbage and yee shall recieve
        allocate_host ();
      }
      val = m_host[i];
    }
    else
    {
      if (!m_host_dirty)
      {
        // host data is valid just return the index
        if (m_host == nullptr)
        {
          std::cout << "get_value with null host ptr: this should not happen\n";
        }
        val = m_host[i];
      }
      else
      {
        // we have to copy a singe value off the gpu
        if (m_device == nullptr)
        {
          // ask for garbage and yee shall recieve
          allocate_device ();
        }
#ifdef ASCENT_USE_CUDA
        cudaMemcpy (&val, &m_device[i], sizeof (T), cudaMemcpyDeviceToHost);
#endif
      }
    }

    return val;
  }

  void set(T *data, const size_t size)
  {
    if (m_host)
    {
      deallocate_host ();
    }
    if (m_device)
    {
      deallocate_device ();
    }
    m_size = size;

    pointer_assignment(data);
  }

  void copy(const T *data, const size_t size)
  {
    if (m_host)
    {
      deallocate_host ();
    }
    if (m_device)
    {
      deallocate_device ();
    }

    m_size = size;
    allocate_host ();
    memcpy (m_host, data, sizeof (T) * m_size);
    m_device_dirty = true;
    m_host_dirty = true;
    m_own_host = true;
  }

  size_t size () const
  {
    return m_size;
  }

  void resize (const size_t size)
  {
    if(!m_own_host)
    {
      ASCENT_ERROR("Array: Cannot resize zero copied array");
    }

    if (size == m_size) return;

    m_host_dirty = true;
    m_device_dirty = true;

    deallocate_host ();
    deallocate_device ();
    m_size = size;
  }

  T *get_device_ptr ()
  {

    if (!m_cuda_enabled)
    {
      return get_host_ptr ();
    }

    if (m_device == nullptr)
    {
      allocate_device ();
    }

    if (m_device_dirty && m_host != nullptr)
    {
      synch_to_device ();
    }

    // indicate that the device has the most recent data
    m_host_dirty = true;
    m_device_dirty = false;
    return m_device;
  }

  const T *get_device_ptr_const ()
  {
    if (!m_cuda_enabled)
    {
      return get_host_ptr ();
    }

    if (m_device == nullptr)
    {
      allocate_device ();
    }

    if (m_device_dirty && m_host != nullptr)
    {
      synch_to_device ();
    }

    m_device_dirty = false;
    return m_device;
  }

  T *get_host_ptr ()
  {
    if (m_host == nullptr)
    {
      allocate_host ();
    }

    if (m_cuda_enabled)
    {
      if (m_host_dirty && m_device != nullptr)
      {
        synch_to_host ();
      }
    }

    // indicate that the host has the most recent data
    m_device_dirty = true;
    m_host_dirty = false;

    return m_host;
  }

  T *get_host_ptr_const ()
  {
    if (m_host == nullptr)
    {
      allocate_host ();
    }

    if (m_cuda_enabled)
    {
      if (m_host_dirty && m_host != nullptr)
      {
        synch_to_host ();
      }
    }

    m_host_dirty = false;

    return m_host;
  }

  void summary ()
  {
    const T *ptr = this->get_host_ptr_const ();
    std::cout << "Array size " << m_size << " :";
    if (m_size > 0)
    {
      const int len = 3;
      int seg1_mx = std::min (size_t (len), m_size);
      for (int i = 0; i < seg1_mx; ++i)
      {
        std::cout << " (" << ptr[i] << ")";
      }
      if (m_size > len)
      {
        std::cout << " ...";
        int seg2_len = std::min (m_size - size_t (len), size_t (len));
        int seg2_str = m_size - seg2_len;
        for (int i = seg2_str; i < m_size; ++i)
        {
          std::cout << " (" << ptr[i] << ")";
        }
      }
    }
    std::cout << "\n";
  }

  void raw_summary ()
  {
    std::cout << "host_ptr = " << m_host << "\n";
    std::cout << "device_ptr = " << m_device << "\n";
  }

  virtual ~ArrayInternals ()
  {
    deallocate_host ();
    deallocate_device ();
  }

  //
  // Allow the release of device memory and save the
  // existing data on the host if applicable
  //
  virtual void release_device_ptr ()
  {
    if (m_cuda_enabled)
    {
      if (m_device != nullptr)
      {

        if (m_host == nullptr)
        {
          allocate_host ();
        }

        if (m_host_dirty)
        {
          synch_to_host ();
        }
      }
    }

    deallocate_device ();
    m_device_dirty = true;
  }

  virtual size_t device_alloc_size ()
  {
    if (m_device == nullptr || !m_own_device )
      return 0;
    else
      return static_cast<size_t> (sizeof (T)) * m_size;
  }

  virtual size_t host_alloc_size ()
  {
    if (m_host == nullptr || !m_own_host)
      return 0;
    else
      return static_cast<size_t> (sizeof (T)) * m_size;
  }

  protected:
  void deallocate_host ()
  {
    if(!m_own_host)
    {
      m_host = nullptr;
      m_host_dirty = true;
    }
    else if (m_host != nullptr)
    {
      auto &rm = umpire::ResourceManager::getInstance ();
      const int allocator_id = AllocationManager::umpire_host_allocator_id();
      umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
      host_allocator.deallocate (m_host);
      m_host = nullptr;
      m_host_dirty = true;
    }
  }

  void allocate_host ()
  {
    if (m_size == 0) return;
    if(!m_own_host)
    {
      ASCENT_ERROR("Array: cannot allocate host when zero copied");
    }

    if (m_host == nullptr)
    {
      auto &rm = umpire::ResourceManager::getInstance ();
      const int allocator_id = AllocationManager::umpire_host_allocator_id();
      umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
      m_host = static_cast<T *> (host_allocator.allocate (m_size * sizeof (T)));
    }
  }

  void deallocate_device ()
  {
    if (m_cuda_enabled)
    {
      if(!m_own_host)
      {
        m_host = nullptr;
        m_host_dirty = true;
      }
      if (m_device != nullptr && m_own_device)
      {
        auto &rm = umpire::ResourceManager::getInstance ();
        const int allocator_id = AllocationManager::umpire_device_allocator_id();
        umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
        //umpire::Allocator device_allocator = rm.getAllocator ("DEVICE");
        device_allocator.deallocate (m_device);
        m_device = nullptr;
        m_device_dirty = true;
      }
    }
  }

  void allocate_device ()
  {
    if (m_size == 0) return;
    if(!m_own_device)
    {
      ASCENT_ERROR("Array: cannot allocate device when zero copied");
    }
    if (m_cuda_enabled)
    {
      if (m_device == nullptr)
      {
        auto &rm = umpire::ResourceManager::getInstance ();
        const int allocator_id = AllocationManager::umpire_device_allocator_id();
        umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
        m_device = static_cast<T *> (device_allocator.allocate (m_size * sizeof (T)));
      }
    }
  }

  // synchs assumes that both arrays are allocated
  void synch_to_host ()
  {
    if(!m_own_host && !m_own_device)
    {
      // this is unified memory, so do nothing
      return;
    }
#ifdef ASCENT_USE_CUDA
    cudaMemcpy(m_host, m_device, m_size * sizeof(T), cudaMemcpyDeviceToHost);
#endif
  }

  void synch_to_device ()
  {
    if(!m_own_host && !m_own_device)
    {
      // this is unified memory, so do nothing
      return;
    }
#ifdef ASCENT_USE_CUDA
    cudaMemcpy(m_device, m_host, m_size * sizeof(T), cudaMemcpyHostToDevice);
#endif
  }

};

} // namespace runtime
} // namespace ascent

#endif
