//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_VTKH_DEVICE_UTILS_HPP
#define ASCENT_VTKH_DEVICE_UTILS_HPP

//-----------------------------------------------------------------------------
///
/// file: ascent_vtkh_device_utils.hpp
///
//-----------------------------------------------------------------------------
#if defined(ASCENT_RAJA_ENABLED)
#include <RAJA/RAJA.hpp>
#endif

#include <vtkm/cont/DataSet.h>

#include <ascent_exports.h>
#include <conduit.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

using index_t = conduit::index_t;

struct EmptyPolicy
{};

#if defined(ASCENT_CUDA_ENABLED)
//---------------------------------------------------------------------------//
// CUDA decorators
//---------------------------------------------------------------------------//
#define ASCENT_EXEC inline __host__ __device__
// Note: there is a performance hit for doing both host and device
// the cuda compiler calls this on then host as a std::function call for each i
// in the for loop, and that basically works out to a virtual function
// call. Thus for small loops, the know overhead is about 3x
#define ASCENT_LAMBDA __device__ __host__
#if defined(ASCENT_RAJA_ENABLED)
#define BLOCK_SIZE 128
using for_policy = RAJA::cuda_exec<BLOCK_SIZE>;
using reduce_policy = RAJA::cuda_reduce;
using atomic_policy = RAJA::cuda_atomic;
#else
using for_policy    = EmptyPolicy;
using reduce_policy = EmptyPolicy;
using atomic_policy = EmptyPolicy;
#endif

#elif defined(ASCENT_HIP_ENABLED) // && ?
//---------------------------------------------------------------------------//
// HIP decorators
//---------------------------------------------------------------------------//
#define ASCENT_EXEC inline __host__ __device__
#define ASCENT_LAMBDA __device__ __host__
#if defined(ASCENT_RAJA_ENABLED)
#define BLOCK_SIZE 256
using for_policy = RAJA::hip_exec<BLOCK_SIZE>;
using reduce_policy = RAJA::hip_reduce;
using atomic_policy = RAJA::hip_atomic;
#else
using for_policy    = EmptyPolicy;
using reduce_policy = EmptyPolicy;
using atomic_policy = EmptyPolicy;
#endif

#else
//---------------------------------------------------------------------------//
// Non-device decorators
//---------------------------------------------------------------------------//
#define ASCENT_EXEC   inline
#define ASCENT_LAMBDA
#if defined(ASCENT_RAJA_ENABLED)
using for_policy = RAJA::seq_exec;
using reduce_policy = RAJA::seq_reduce;
using atomic_policy = RAJA::seq_atomic;
#else
using for_policy    = EmptyPolicy;
#endif
#endif



//---------------------------------------------------------------------------//
// Device Error Checks
//---------------------------------------------------------------------------//
#if defined(ASCENT_CUDA_ENABLED)
//---------------------------------------------------------------------------//
// cuda error check
//---------------------------------------------------------------------------//
inline void cuda_error_check(const char *file, const int line )
{
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    std::cerr<<"CUDA error reported at: "<<file<<":"<<line;
    std::cerr<<" : "<<cudaGetErrorString(err)<<"\n";
    //exit( -1 );
  }
}
#define ASCENT_DEVICE_ERROR_CHECK() cuda_error_check(__FILE__,__LINE__);

#elif defined(ASCENT_HIP_ENABLED)
//---------------------------------------------------------------------------//
// hip error check
//---------------------------------------------------------------------------//
inline void hip_error_check(const char *file, const int line )
{
  hipError_t err = hipGetLastError();
  if ( hipSuccess != err )
  {
    std::cerr<<"HIP error reported at: "<<file<<":"<<line;
    std::cerr<<" : "<<hipGetErrorName(err)<<"\n";
    //exit( -1 );
  }
}
#define ASCENT_DEVICE_ERROR_CHECK() hip_error_check(__FILE__,__LINE__);
#else
//---------------------------------------------------------------------------//
// non-device error check (no op)
//---------------------------------------------------------------------------//
#define ASCENT_DEVICE_ERROR_CHECK()
#endif

//---------------------------------------------------------------------------//
// forall
//---------------------------------------------------------------------------//
#if defined(ASCENT_RAJA_ENABLED)
template <typename ExecPolicy, typename Kernel>
inline void forall(const index_t& begin,
                   const index_t& end,
                   Kernel&& kernel) noexcept
{
    RAJA::forall<ExecPolicy>(RAJA::RangeSegment(begin, end),
                             std::forward<Kernel>(kernel));
}
#else
template <typename ExecPolicy, typename Kernel>
inline void forall(const index_t& begin,
                   const index_t& end,
                   Kernel&& kernel) noexcept
{
    for(index_t i = begin; i < end; ++i)
    {
        kernel(i);
    };
}
#endif

//-----------------------------------------------------------------------------
// -- start VTKHDeviceAdapter --
//-----------------------------------------------------------------------------
class ASCENT_API VTKHDeviceAdapter
{

public: 
  // Definition of the cast function with __host__ __device__
  template <typename T, typename S> ASCENT_EXEC static void castUint64ToFloat64(const T* input, S* output, int size)
  {
      forall<for_policy>(0, size, [=] ASCENT_LAMBDA(index_t i)
      {
          output[i] = static_cast<S>(input[i]);
      });
      ASCENT_DEVICE_ERROR_CHECK();
  }

//-----------------------------------------------------------------------------
// -- end VTKHDeviceAdapter --
//-----------------------------------------------------------------------------
};

//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end header --
//-----------------------------------------------------------------------------
#endif
