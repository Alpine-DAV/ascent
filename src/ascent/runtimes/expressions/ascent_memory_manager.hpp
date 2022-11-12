#ifndef ASCENT_MEMORY_MANAGER
#define ASCENT_MEMORY_MANAGER

#include <stddef.h>
#include <conduit.hpp>
#include <ascent_exports.h>

namespace ascent
{
///
/// Interfaces for host and device memory allocation / deallocation.
///


//-----------------------------------------------------------------------------
/// Interface to set allocator ids (singleton)
//-----------------------------------------------------------------------------
class ASCENT_API AllocationManager
{
public:
  /// Return host allocator id
  ///  If Umpire is enabled and no allocator has been set,
  ///  an Umpire "HOST_POOL" allocator is created, set, and returned.
  static int  host_allocator_id();

  /// Return device allocator id
  ///  If Umpire is enabled and no allocator has been set,
  ///  an Umpire "GPU_POOL" allocator is created, set, and returned.
  /// If Umpire is disabled, an error is thrown
  static int  device_allocator_id();

  /// set umpire host allocator from outside ascent via id
  /// Throws an error if Umpire is disabled
  static bool set_host_allocator_id(int id);

  /// set umpire device allocator from outside ascent via id
  /// Throws an error if Umpire is disabled
  static bool set_device_allocator_id(int id);

  // registered conduit magic memory allocator id for host memory
  static int  conduit_host_allocator_id();
  // registered conduit magic memory allocator id for device memory
  static int  conduit_device_allocator_id();

  // registers the fancy conduit memory handlers for
  // magic memset and memcpy
  static void set_conduit_mem_handlers();

private:
  static int  m_host_allocator_id;
  static int  m_device_allocator_id;
  
  static int  m_conduit_host_allocator_id;
  static int  m_conduit_device_allocator_id;

  static bool m_external_host_allocator;
  static bool m_external_device_allocator;

};

//-----------------------------------------------------------------------------
/// Host Memory allocation / deallocation interface (singleton)
///  Uses AllocationManager::host_allocator_id() when Umpire is enabled,
///  Uses malloc/free when Umpire is disabled. 
//-----------------------------------------------------------------------------
struct ASCENT_API HostMemory
{
  static void *allocate(size_t bytes);
  static void *allocate(size_t items, size_t item_size);
  static void  deallocate(void *data_ptr);

private:
  static size_t m_total_bytes_alloced;
  static size_t m_alloc_count;
  static size_t m_free_count;

};
//-----------------------------------------------------------------------------
/// Device Memory allocation / deallocation interface (singleton)
///  Uses AllocationManager::device_allocator_id() when Umpire is enabled.
///  allocate() and deallocate() throw errors when Umpire is disabled.
//-----------------------------------------------------------------------------
struct ASCENT_API DeviceMemory
{
  static void *allocate(size_t bytes);
  static void *allocate(size_t items, size_t item_size);
  static void  deallocate(void *data_ptr);

  static bool is_device_ptr(const void *ptr);
  static void is_device_ptr(const void *ptr, bool &is_gpu, bool &is_unified);

private:
  static size_t m_total_bytes_alloced;
  static size_t m_alloc_count;
  static size_t m_free_count;

};

//-----------------------------------------------------------------------------
struct ASCENT_API MagicMemory
{
  static void memset(void *ptr, int value, size_t num );
  static void copy(void *destination, const void *source, size_t num);
};


} // namespace ascent
#endif
