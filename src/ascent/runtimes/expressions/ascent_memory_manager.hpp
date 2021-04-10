#ifndef ASCENT_MEMORY_MANAGER
#define ASCENT_MEMORY_MANAGER

#include <stddef.h>
#include <conduit.hpp>

namespace ascent
{

bool is_gpu_ptr(const void *ptr);
void is_gpu_ptr(const void *ptr, bool &is_gpu, bool &is_unified);

class AllocationManager
{
public:

  static int umpire_device_allocator_id();
  // set a device allocator from outside ascent
  static bool umpire_device_allocator_id(int id);

  static int umpire_host_allocator_id();
  static int conduit_host_allocator_id();
  static int conduit_device_allocator_id();
  // set the fancy conduit memory handlers for
  // magic memset and memcpy
  static void set_conduit_mem_handlers();

private:
  static int m_umpire_device_allocator_id;
  static int m_umpire_host_allocator_id;
  static int m_conduit_host_allocator_id;
  static int m_conduit_device_allocator_id;
  static bool m_external_device_allocator;
};

struct HostAllocator
{
  static size_t m_total_bytes_alloced;
  static size_t m_alloc_count;
  static size_t m_free_count;

  static void * alloc(size_t items, size_t item_size);
  static void free(void *data_ptr);
};

struct MagicMemory
{
  static void memset(void * ptr, int value, size_t num );
  static void copy(void * destination, const void * source, size_t num);
};

struct DeviceAllocator
{
  static size_t m_total_bytes_alloced;
  static size_t m_alloc_count;
  static size_t m_free_count;

  static void * alloc(size_t items, size_t item_size);
  static void free(void *data_ptr);
};

} // namespace ascent
#endif
