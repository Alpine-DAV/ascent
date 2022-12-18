//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_raja_smoke.cpp
///
//-----------------------------------------------------------------------------
#include "gtest/gtest.h"
#include <RAJA/RAJA.hpp>
#include <iostream>



//-----------------------------------------------------------------------------
// device decs
//-----------------------------------------------------------------------------
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)

#define EXEC_LAMBDA __device__ __host__

#else

#define EXEC_LAMBDA 

#endif

//-----------------------------------------------------------------------------
// memory helpers
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void *
device_alloc(int size)
{
#if defined (RAJA_ENABLE_CUDA)
    void *buff;
    cudaMalloc(&buff, size);
    return buff;
#elif defined (RAJA_ENABLE_HIP)
    void *buff;
    hipMalloc(&buff, size);
    return buff;
#else
    return malloc(size);
#endif
}

//-----------------------------------------------------------------------------
void
device_free(void *ptr)

{
#if defined (RAJA_ENABLE_CUDA)
    cudaFree(ptr);
#elif defined (RAJA_ENABLE_HIP)
    hipFree(ptr);
#else
    free(ptr);
#endif
}

//-----------------------------------------------------------------------------
void
copy_from_device_to_host(void *dest, void *src, int size)
{
#if defined (RAJA_ENABLE_CUDA)
   cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
#elif defined (RAJA_ENABLE_HIP)
   hipMemcpy(dest, src, size, hipMemcpyDeviceToHost);
#else
   memcpy(dest,src,size);
#endif
}


//-----------------------------------------------------------------------------
TEST(raja_smoke, basic_use_default_policy)
{
// setup exec policy
#if defined (RAJA_ENABLE_OPENMP)
    using ExecPolicy = RAJA::omp_parallel_for_exec;
#elif defined (RAJA_ENABLE_CUDA)
    #define CUDA_BLOCK_SIZE 128
    using ExecPolicy = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
#elif defined (RAJA_ENABLE_HIP)
    #define HIP_BLOCK_SIZE 256
    using ExecPolicy = RAJA::hip_exec<HIP_BLOCK_SIZE>;
#else
    using ExecPolicy = RAJA::seq_exec;
#endif

   const int size = 10;

   int host_vals[size];
   int *dev_vals_ptr = static_cast<int*>(device_alloc(sizeof(int) * size));

   RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, size), [=] EXEC_LAMBDA (int i) {
        dev_vals_ptr[i] = i;
   });

   copy_from_device_to_host( &host_vals[0]  , dev_vals_ptr,  sizeof(int) * size);

   for(int i=0;i<size;i++)
   {
      EXPECT_EQ(host_vals[i],i);
   }

   device_free(dev_vals_ptr);
}



