#include "ascent_memory_manager.hpp"
#include <ascent_logging.hpp>
#include <ascent_config.h>


#if defined(ASCENT_UMPIRE_ENABLED)
#include <umpire/Umpire.hpp>
#include <umpire/util/MemoryResourceTraits.hpp>
#include <umpire/strategy/DynamicPoolList.hpp>
#endif
#include <cstring> // memcpy
#include <conduit.hpp>

namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Allocation Manager
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

int  AllocationManager::m_host_allocator_id  = -1;
int  AllocationManager::m_device_allocator_id = -1;

int  AllocationManager::m_conduit_host_allocator_id   = -1;
int  AllocationManager::m_conduit_device_allocator_id = -1;

bool AllocationManager::m_external_host_allocator = false;
bool AllocationManager::m_external_device_allocator = false;

//-----------------------------------------------------------------------------
int
AllocationManager::host_allocator_id()
{
  if(m_host_allocator_id == -1)
  {
#if !defined(ASCENT_UMPIRE_ENABLED)
         ASCENT_ERROR("Ascent was built without Umpire Support. "
                       "Cannot access host allocator id");
#else
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator("HOST");
    // we can use the umpire profiling to find a good default size
    auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
                            "HOST_POOL",
                            allocator,
                            1ul * // 1GB default size
                            1024ul * 1024ul * 1024ul + 1);
    m_host_allocator_id = pooled_allocator.getId();
#endif
  }
  return m_host_allocator_id;
}

//-----------------------------------------------------------------------------
int
AllocationManager::device_allocator_id()
{
  if(m_device_allocator_id == -1)
  {
#if !defined(ASCENT_UMPIRE_ENABLED)
         ASCENT_ERROR("Ascent was built without Umpire Support. "
                       "Cannot access device allocator id");
#else
    auto &rm = umpire::ResourceManager::getInstance ();
    auto allocator = rm.getAllocator("DEVICE");
    // we can use the umpire profiling to find a good default size

    auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
                            "GPU_POOL",
                            allocator,
                            1ul * // 1GB default size
                            1024ul * 1024ul * 1024ul + 1);
    m_device_allocator_id = pooled_allocator.getId();
#endif
  }

  return m_device_allocator_id;
}

//-----------------------------------------------------------------------------
bool
AllocationManager::set_host_allocator_id(int id)
{
    if(m_external_host_allocator && m_host_allocator_id != id)
    {
        // We can't change allocators mid stream.
        // This would cause a mismatch between memory allocated with one 'allocator' then that
        // memory being deallocated with another.
        ASCENT_ERROR("Changing the host allocator id in the middle of a run is not supported.");
    }

#if !defined(ASCENT_UMPIRE_ENABLED)
    ASCENT_ERROR("Ascent was built without Umpire Support. "
                 "Cannot set host allocator id.");
#else

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
#endif
}



//-----------------------------------------------------------------------------
bool
AllocationManager::set_device_allocator_id(int id)
{
    if(m_external_device_allocator && m_device_allocator_id != id)
    {
        // We can't change allocators mid stream.
        // This would cause a mismatch between memory allocated with one 'allocator' then that
        // memory being deallocated with another.
        ASCENT_ERROR("Changing the device allocator id in the middle of a run is not supported.");
    }

#if !defined(ASCENT_UMPIRE_ENABLED)
    ASCENT_ERROR("Ascent was built without Umpire Support. "
                 "Cannot set device allocator id.");
#else
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

#if defined(ASCENT_DEVICE_ENABLED)
    need_device = true;
#endif

    bool is_device = resource == umpire::MemoryResourceTraits::resource_type::device;
    bool is_host   = resource == umpire::MemoryResourceTraits::resource_type::host;

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

    m_device_allocator_id = id;
    m_external_device_allocator = true;

    return true;
#endif
}

//-----------------------------------------------------------------------------
int
AllocationManager::conduit_host_allocator_id()
{
  if(m_conduit_host_allocator_id == -1)
  {
    m_conduit_host_allocator_id
      = conduit::utils::register_allocator(HostMemory::allocate,
                                           HostMemory::deallocate);

    //std::cout<<"Created host allocator "<<m_conduit_host_allocator_id<<"\n";
  }
  //std::cout<<"conduit host allocator "<<m_conduit_host_allocator_id<<"\n";
  return m_conduit_host_allocator_id;
}

//-----------------------------------------------------------------------------
int
AllocationManager::conduit_device_allocator_id()
{
  if(m_conduit_device_allocator_id == -1)
  {
    m_conduit_device_allocator_id
      = conduit::utils::register_allocator(DeviceMemory::allocate,
                                           DeviceMemory::deallocate);

    //std::cout<<"Created device allocator "<<m_conduit_device_allocator_id<<"\n";
  }
  return m_conduit_device_allocator_id;
}

//-----------------------------------------------------------------------------
void
AllocationManager::set_conduit_mem_handlers()
{
#if defined(ASCENT_DEVICE_ENABLED)
  // we only need to override the mem handlers in the
  // presence of cuda or hip
  conduit::utils::set_memcpy_handler(MagicMemory::copy);
  conduit::utils::set_memset_handler(MagicMemory::memset);
#endif
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Host Memory
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
size_t HostMemory::m_total_bytes_alloced = 0;
size_t HostMemory::m_alloc_count = 0;
size_t HostMemory::m_free_count = 0;

//-----------------------------------------------------------------------------
void *
HostMemory::allocate(size_t bytes)
{
  m_total_bytes_alloced += bytes;
  m_alloc_count++;
#if defined(ASCENT_UMPIRE_ENABLED)
  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::host_allocator_id();
  umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
  return host_allocator.allocate(bytes);
#else
  return malloc(bytes);
#endif
}


//-----------------------------------------------------------------------------
void *
HostMemory::allocate(size_t items, size_t item_size)
{
  return allocate(items * item_size);
}

//-----------------------------------------------------------------------------
void
HostMemory::deallocate(void *data_ptr)
{
  m_free_count++;
#if defined(ASCENT_UMPIRE_ENABLED)
  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::host_allocator_id();
  umpire::Allocator host_allocator = rm.getAllocator (allocator_id);
  host_allocator.deallocate(data_ptr);
#else
  return free(data_ptr);
#endif
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Device Memory
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
size_t DeviceMemory::m_total_bytes_alloced = 0;
size_t DeviceMemory::m_alloc_count = 0;
size_t DeviceMemory::m_free_count = 0;

//-----------------------------------------------------------------------------
void *
DeviceMemory::allocate(size_t bytes)
{
#if !defined(ASCENT_UMPIRE_ENABLED)
     ASCENT_ERROR("Ascent was built without Umpire support. "
                  "Cannot use DeviceMemory::alloc().");
#endif

#if defined(ASCENT_DEVICE_ENABLED)
  m_total_bytes_alloced += bytes;
  m_alloc_count++;
  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::device_allocator_id();
  umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
  return device_allocator.allocate(bytes);
#else
  (void) bytes; // unused
  ASCENT_ERROR("Calling device allocator when no device is present.");
  return nullptr;
#endif
}

//-----------------------------------------------------------------------------
void *
DeviceMemory::allocate(size_t items, size_t item_size)
{
    return allocate(items * item_size);
}

//-----------------------------------------------------------------------------
void
DeviceMemory::deallocate(void *data_ptr)
{
#if !defined(ASCENT_UMPIRE_ENABLED)
     ASCENT_ERROR("Ascent was built without Umpire support. "
                  "Cannot use DeviceMemory::free().");
#endif

#if defined(ASCENT_DEVICE_ENABLED)
  m_free_count++;
  auto &rm = umpire::ResourceManager::getInstance ();
  const int allocator_id = AllocationManager::device_allocator_id();
  umpire::Allocator device_allocator = rm.getAllocator (allocator_id);
  device_allocator.deallocate (data_ptr);
#else
  (void) data_ptr;
  ASCENT_ERROR("Calling device allocator when no device is present.");
#endif
}


//-----------------------------------------------------------------------------
void
DeviceMemory::is_device_ptr(const void *ptr, bool &is_gpu, bool &is_unified)
{
    is_gpu = false;
    is_unified = false;
#if defined(ASCENT_CUDA_ENABLED)
    cudaPointerAttributes atts;
    const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);

    is_gpu = false;
    is_unified = false;

    // clear last error so other error checking does
    // not pick it up
    cudaError_t error = cudaGetLastError();
    is_gpu = (perr == cudaSuccess) &&
             (atts.type == cudaMemoryTypeDevice ||
              atts.type == cudaMemoryTypeManaged   );

    is_unified = cudaSuccess && atts.type == cudaMemoryTypeDevice;
#elif defined(ASCENT_HIP_ENABLED)
    hipPointerAttributes_t atts;
    const hipError_t perr = hipPointerGetAttributes(&atts, ptr);

    is_gpu = false;
    is_unified = false;

    // clear last error so other error checking does
    // not pick it up
    hipError_t perr = hipGetLastError();
    is_gpu = (perr == hipSuccess) &&
             (atts.type == hipMemoryTypeDevice ||
              atts.type ==  hipMemoryTypeManaged );
    is_unified = (hipSuccess && atts.type == hipMemoryTypeDevice);
#endif 
}

//-----------------------------------------------------------------------------
// Adapted from:
// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool
DeviceMemory::is_device_ptr(const void *ptr)
{
#if defined(ASCENT_CUDA_ENABLED)
    cudaPointerAttributes atts;
    const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);
    // clear last error so other error checking does
    // not pick it up
    cudaError_t error = cudaGetLastError();
    return perr == cudaSuccess &&
                (atts.type == cudaMemoryTypeDevice ||
                 atts.type == cudaMemoryTypeManaged);

#elif defined(ASCENT_HIP_ENABLED)
    hipPointerAttributes_t atts;
    const hipError_t perr = cudaPointerGetAttributes(&atts, ptr);
    // clear last error so other error checking does
    // not pick it up
    hipError_t error = cudaGetLastError();
    return perr == hipSuccess &&
                (atts.type == hipMemoryTypeDevice ||
                 atts.type == hipMemoryTypeManaged);
#else
  (void) ptr;
  return false;
#endif
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Magic Memory
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
MagicMemory::memset(void * ptr, int value, size_t num )
{
#if defined(ASCENT_DEVICE_ENABLED)
  bool is_device = DeviceMemory::is_device_ptr(ptr);
  if(is_device)
  {
    #if defined(ASCENT_CUDA_ENABLED)
        cudaMemset(ptr,value,num);
    #elif defined(ASCENT_HIP_ENABLED)
        hipMemset(ptr,value,num);
    #endif
  }
  else
  {
    memset(ptr,value,num);
  }
#else
  memset(ptr,value,num);
#endif
}

//-----------------------------------------------------------------------------
void
MagicMemory::copy(void * destination, const void * source, size_t num)
{
#if defined(ASCENT_DEVICE_ENABLED)
  bool src_is_gpu = DeviceMemory::is_device_ptr(source);
  bool dst_is_gpu = DeviceMemory::is_device_ptr(destination);
  if(src_is_gpu && dst_is_gpu)
  {
     #if defined(ASCENT_CUDA_ENABLED)
         cudaMemcpy(destination, source, num, cudaMemcpyDeviceToDevice);
     #elif defined(ASCENT_HIP_ENABLED)
         hipMemcpy(destination, source, num, hipMemcpyDeviceToDevice);
     #endif
  }
  else if(src_is_gpu && !dst_is_gpu)
  {
    #if defined(ASCENT_CUDA_ENABLED)
        cudaMemcpy(destination, source, num, cudaMemcpyDeviceToHost);
    #elif defined(ASCENT_HIP_ENABLED)
        hipMemcpy(destination, source, num, hipMemcpyDeviceToHost);
    #endif
  }
  else if(!src_is_gpu && dst_is_gpu)
  {
    #if defined(ASCENT_CUDA_ENABLED)
        cudaMemcpy(destination, source, num, cudaMemcpyHostToDevice);
    #elif defined(ASCENT_HIP_ENABLED)
        hipMemcpy(destination, source, num, hipMemcpyHostToDevice);
    #endif
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
