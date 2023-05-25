// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ARRAY_INTERNALS
#define DRAY_ARRAY_INTERNALS

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array_internals_base.hpp>
#include <dray/array_registry.hpp>
#include <dray/math.hpp>

#include <umpire/Umpire.hpp>

#include <iostream>
#include <string.h>

namespace dray
{

template <typename T> class ArrayInternals : public ArrayInternalsBase
{
  protected:
  T *m_device;
  T *m_host;
  bool m_device_dirty;
  bool m_host_dirty;
  size_t m_size;
  bool m_cuda_enabled;
  bool m_hip_enabled;

  public:
  ArrayInternals ()
  : ArrayInternalsBase (), m_device (nullptr), m_host (nullptr),
    m_device_dirty (true), m_host_dirty (true), m_size (0)
  {
#ifdef DRAY_CUDA_ENABLED
    m_cuda_enabled = true;
#else
    m_cuda_enabled = false;
#endif
#ifdef DRAY_HIP_ENABLED
    m_hip_enabled = true;
#else
    m_hip_enabled = false;
#endif
  }

  ArrayInternals (const T *data, const int32 size)
  : ArrayInternalsBase (), m_device (nullptr), m_host (nullptr),
    m_device_dirty (true), m_host_dirty (false), m_size (size)
  {
#ifdef DRAY_CUDA_ENABLED
    m_cuda_enabled = true;
#else
    m_cuda_enabled = false;
#endif
#ifdef DRAY_HIP_ENABLED
    m_hip_enabled = true;
#else
    m_hip_enabled = false;
#endif
    allocate_host ();
    // copy data in
    memcpy (m_host, data, sizeof (T) * m_size);
  }

  T get_value (const int32 i)
  {
    assert (i >= 0);
    assert (i < m_size);
    T val = T();
    // host only
    if (!m_cuda_enabled && !m_hip_enabled)
    {
      if (m_host == nullptr)
      {
        // ask for garbage and yee shall recieve
        allocate_host ();
      }
      val = m_host[i];
    }
    // HIP
    else if(m_hip_enabled)
    {
      if (!m_host_dirty)
      {
        // host data is valud just return the index
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
#ifdef DRAY_HIP_ENABLED
        hipMemcpy (&val, &m_device[i], sizeof (T), hipMemcpyDeviceToHost);
#endif
      }
    }
    // CUDA
    else if(m_cuda_enabled)
    {
      if (!m_host_dirty)
      {
        // host data is valud just return the index
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
          // ask for garbage and yee shall receive
          allocate_device ();
        }
#ifdef DRAY_CUDA_ENABLED
        cudaMemcpy (&val, &m_device[i], sizeof (T), cudaMemcpyDeviceToHost);
#endif
      }
    }

    return val;
  }

  void set (const T *data, const int32 size)
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
  }

  size_t size () const
  {
    return m_size;
  }

  void resize (const size_t size)
  {

    if (size == m_size) return;

    m_host_dirty = true;
    m_device_dirty = true;

    deallocate_host ();
    deallocate_device ();
    m_size = size;
  }

  T *get_device_ptr ()
  {

    if (!m_cuda_enabled && !m_hip_enabled)
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
    if (!m_cuda_enabled && !m_hip_enabled)
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
    else if(m_hip_enabled)
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
    else if (m_hip_enabled)
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
      int32 seg1_mx = std::min (size_t (len), m_size);
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

  virtual ~ArrayInternals () override
  {
    deallocate_host ();
    deallocate_device ();
  }

  //
  // Allow the release of device memory and save the
  // existing data on the host if applicable
  //
  virtual void release_device_ptr () override
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

  virtual size_t device_alloc_size () override
  {
    if (m_device == nullptr)
      return 0;
    else
      return static_cast<size_t> (sizeof (T)) * m_size;
  }

  virtual size_t host_alloc_size () override
  {
    if (m_host == nullptr)
      return 0;
    else
      return static_cast<size_t> (sizeof (T)) * m_size;
  }

  protected:
  void deallocate_host ()
  {
    if (m_host != nullptr)
    {
      auto &rm = umpire::ResourceManager::getInstance ();
      const int allocator_id = ArrayRegistry::host_allocator_id();
      umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
      host_allocator.deallocate (m_host);
      m_host = nullptr;
      m_host_dirty = true;
    }
  }

  void allocate_host ()
  {
    if (m_size == 0) return;
    if (m_host == nullptr)
    {
      auto &rm = umpire::ResourceManager::getInstance ();
      const int allocator_id = ArrayRegistry::host_allocator_id();
      umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
      m_host = static_cast<T *> (host_allocator.allocate (m_size * sizeof (T)));
    }
  }

  void deallocate_device ()
  {
    if (m_cuda_enabled || m_hip_enabled)
    {
      if (m_device != nullptr)
      {
        auto &rm = umpire::ResourceManager::getInstance ();
        const int allocator_id = ArrayRegistry::device_allocator_id();
        umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
        device_allocator.deallocate (m_device);
        m_device = nullptr;
        m_device_dirty = true;
      }
    }
  }

  void allocate_device ()
  {
    if (m_size == 0) return;
    if (m_cuda_enabled || m_hip_enabled)
    {
      if (m_device == nullptr)
      {
        auto &rm = umpire::ResourceManager::getInstance ();
        const int allocator_id = ArrayRegistry::device_allocator_id();
        umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
        m_device = static_cast<T *> (device_allocator.allocate (m_size * sizeof (T)));
      }
    }
  }

  // synchs assumes that both arrays are allocated
  void synch_to_host ()
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    rm.copy (m_host, m_device);
  }

  void synch_to_device ()
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    rm.copy (m_device, m_host);
  }
};

} // namespace dray

#endif
