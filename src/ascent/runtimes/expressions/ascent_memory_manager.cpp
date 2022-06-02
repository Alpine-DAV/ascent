#include "ascent_memory_manager.hpp"
#include <ascent_logging.hpp>
#include <ascent_config.h>

#include <umpire/Umpire.hpp>
#include <umpire/util/MemoryResourceTraits.hpp>
#include <umpire/strategy/DynamicPoolList.hpp>

#include <conduit.hpp>

namespace ascent
{
  //
void is_gpu_ptr(const void *ptr, bool &is_gpu, bool &is_unified)
{
  is_gpu = false;
  is_unified = false;
#ifdef ASCENT_USE_CUDA
  cudaPointerAttributes atts;
  const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);

  is_gpu = false;
  is_unified = false;
  // clear last error so other error checking does
  // not pick it up
  cudaError_t error = cudaGetLastError();
#if CUDART_VERSION >= 10000
  is_gpu = perr == cudaSuccess &&
                   (atts.type == cudaMemoryTypeDevice ||
                   atts.type == cudaMemoryTypeManaged);
  is_unified = cudaSuccess && atts.type == cudaMemoryTypeDevice;
#else
  is_gpu = perr == cudaSuccess && atts.memoryType == cudaMemoryTypeDevice;
  is_unified = false;
#endif
  // This will gen an error when the pointer is not a GPU pointer.
  // Clear the error so others don't pick it up.
  error = cudaGetLastError();
  (void) error;
#endif
}

// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool is_gpu_ptr(const void *ptr)
{
#ifdef ASCENT_USE_CUDA
  cudaPointerAttributes atts;
  const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);

  // clear last error so other error checking does
  // not pick it up
  cudaError_t error = cudaGetLastError();
  #if CUDART_VERSION >= 10000
  return perr == cudaSuccess &&
                (atts.type == cudaMemoryTypeDevice ||
                 atts.type == cudaMemoryTypeManaged);
  #else
  return perr == cudaSuccess && atts.memoryType == cudaMemoryTypeDevice;
  #endif
#else
  (void) ptr;
  return false;
#endif
}

int AllocationManager::m_umpire_device_allocator_id = -1;
int AllocationManager::m_umpire_host_allocator_id = -1;
int AllocationManager::m_conduit_host_allocator_id = -1;
int AllocationManager::m_conduit_device_allocator_id = -1;
bool AllocationManager::m_external_device_allocator = false;

int
AllocationManager::umpire_device_allocator_id()
{
  if(m_umpire_device_allocator_id == -1)
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator("DEVICE");
    // we can use the umpire profiling to find a good default size

    auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
                            "GPU_POOL",
                            allocator,
                            1ul * // 1GB default size
                            1024ul * 1024ul * 1024ul + 1);
    m_umpire_device_allocator_id = pooled_allocator.getId();
  }
  return m_umpire_device_allocator_id;
}

int
AllocationManager::umpire_host_allocator_id()
{
  if(m_umpire_host_allocator_id == -1)
  {
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator("HOST");
    // we can use the umpire profiling to find a good default size
    auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
                            "HOST_POOL",
                            allocator,
                            1ul * // 1GB default size
                            1024ul * 1024ul * 1024ul + 1);
    m_umpire_host_allocator_id = pooled_allocator.getId();
  }
  return m_umpire_host_allocator_id;
}

bool
AllocationManager::umpire_device_allocator_id(int id)
{

  if(m_external_device_allocator && m_umpire_device_allocator_id != id)
  {
    // with the current implementation, i cant control switching allocators in the middle
    // This would cause a mismatch between memory allocated with one 'allocator' then that
    // memory being deallocated with another. Something to think about
    ASCENT_ERROR("Setting the device allocator id in the middle of the run to something new is BAD\n");
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

  if(!valid_id)
  {
    return false;
  }

  auto resource = allocator.getAllocationStrategy()->getTraits().resource;

  bool can_use = false;
  bool need_device = false;

#ifdef ASCENT_USE_CUDA
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

  if(id != m_umpire_device_allocator_id)
  {
    // Matt: i don't think anyone will s
    // if this is not the same, we have to get rid
    // of all currently allocated deviec resources.
    // Data will be preserved by a synch to host
    //release_device_res();
    m_umpire_device_allocator_id = id;
  }
  m_external_device_allocator = true;
  return true;
}

int
AllocationManager::conduit_host_allocator_id()
{
  if(m_conduit_host_allocator_id == -1)
  {
    m_conduit_host_allocator_id
      = conduit::utils::register_allocator(HostAllocator::alloc,
                                           HostAllocator::free);

    std::cout<<"Created host allocator "<<m_conduit_host_allocator_id<<"\n";
  }
  std::cout<<"conduit host allocator "<<m_conduit_host_allocator_id<<"\n";
  return m_conduit_host_allocator_id;
}

int
AllocationManager::conduit_device_allocator_id()
{
  if(m_conduit_device_allocator_id == -1)
  {
    m_conduit_device_allocator_id
      = conduit::utils::register_allocator(DeviceAllocator::alloc,
                                             DeviceAllocator::free);

    std::cout<<"Created device allocator "<<m_conduit_device_allocator_id<<"\n";
  }
  return m_conduit_device_allocator_id;
}

void AllocationManager::set_conduit_mem_handlers()
{
#ifdef ASCENT_USE_CUDA
  // we only need to overide the mem handlers in the
  // presence of cuda
  conduit::utils::set_memcpy_handler(MagicMemory::copy);
  conduit::utils::set_memset_handler(MagicMemory::memset);
#endif
}

// ------------------------- Host Allocator -----------------------------------
size_t HostAllocator::m_total_bytes_alloced = 0;
size_t HostAllocator::m_alloc_count = 0;
size_t HostAllocator::m_free_count = 0;

void *
HostAllocator::alloc(size_t items, size_t item_size)
{
  std::cout<<"Bananas allocate\n";
  m_total_bytes_alloced += items * item_size;
  m_alloc_count++;
  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::umpire_host_allocator_id();
  umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
  return host_allocator.allocate (items * item_size);
}

void
HostAllocator::free(void *data_ptr)
{
  std::cout<<"free bananas\n";
  m_free_count++;

  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::umpire_host_allocator_id();
  umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
  host_allocator.deallocate (data_ptr);
}

// ------------------------- Host Allocator -----------------------------------
size_t DeviceAllocator::m_total_bytes_alloced = 0;
size_t DeviceAllocator::m_alloc_count = 0;
size_t DeviceAllocator::m_free_count = 0;

void *
DeviceAllocator::alloc(size_t items, size_t item_size)
{
#ifdef ASCENT_USE_CUDA
  m_total_bytes_alloced += items * item_size;
  m_alloc_count++;
  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::umpire_device_allocator_id();
  umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
  return device_allocator.allocate (items * item_size);
#else
  (void) items;
  (void) item_size;
  ASCENT_ERROR("Calling device allocator when no device is present.");
  return nullptr;
#endif
}

void
DeviceAllocator::free(void *data_ptr)
{
#ifdef ASCENT_USE_CUDA
  m_free_count++;
  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::umpire_device_allocator_id();
  umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
  device_allocator.deallocate (data_ptr);
#else
  (void) data_ptr;
  ASCENT_ERROR("Calling device allocator when no device is present.");
#endif
}

void
MagicMemory::memset(void * ptr, int value, size_t num )
{
#ifdef ASCENT_USE_CUDA
  bool is_device = is_gpu_ptr(ptr);
  if(is_device)
  {
    cudaMemset(ptr,value,num);
  }
  else
  {
    memset(ptr,value,num);
  }
#else
  memset(ptr,value,num);
#endif
}

void
MagicMemory::copy(void * destination, const void * source, size_t num)
{
#ifdef ASCENT_USE_CUDA
  bool src_is_gpu = is_gpu_ptr(source);
  bool dst_is_gpu = is_gpu_ptr(destination);
  if(src_is_gpu && dst_is_gpu)
  {
    cudaMemcpy(destination, source, num, cudaMemcpyDeviceToDevice);
  }
  else if(src_is_gpu && !dst_is_gpu)
  {
    cudaMemcpy(destination, source, num, cudaMemcpyDeviceToHost);
  }
  else if(!src_is_gpu && dst_is_gpu)
  {
    cudaMemcpy(destination, source, num, cudaMemcpyHostToDevice);
  }
  else
  {
    // we are the default memcpy in conduit so this is the normal
    // path
    memcpy(destination,source,num);
  }
#else
  memcpy(destination,source,num);
#endif
}

} // namespace ascent
