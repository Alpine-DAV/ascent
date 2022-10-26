#ifndef ASCENT_EXECUTION_POLICIES_HPP
#define ASCENT_EXECUTION_POLICIES_HPP

#include <ascent_config.h>
#include <conduit.hpp>

#if defined(ASCENT_RAJA_ENABLED)
#include <RAJA/RAJA.hpp>
#endif

namespace ascent
{

//---------------------------------------------------------------------------//
#if defined(ASCENT_RAJA_ENABLED)
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// policies for when raja is on
//---------------------------------------------------------------------------//

#if defined(ASCENT_CUDA_ENABLED)
#define CUDA_BLOCK_SIZE 128
#endif

#if defined(ASCENT_HIP_ENABLED)
#define HIP_BLOCK_SIZE 256
#endif

//---------------------------------------------------------------------------//
#if defined(ASCENT_CUDA_ENABLED)
struct CudaExec
{
  using for_policy    = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using reduce_policy = RAJA::cuda_reduce;
  using atomic_policy = RAJA::cuda_atomic;
  static std::string memory_space;
};
#endif

//---------------------------------------------------------------------------//
#if defined(ASCENT_HIP_ENABLED)
struct HipExec
{
  using for_policy    = RAJA::hip_exec<HIP_BLOCK_SIZE>;
  using reduce_policy = RAJA::hip_reduce;
  using atomic_policy = RAJA::hip_atomic;
  static std::string memory_space;
};
#endif

//---------------------------------------------------------------------------//
#if defined(ASCENT_OPENMP_ENABLED)
struct OpenMPExec
{
  using for_policy = RAJA::omp_parallel_for_exec;
#if defined(ASCENT_CUDA_ENABLE)
  // the cuda policy for reductions can be used
  // by other backends, and this should suppress
  // erroneous host device warnings
  using reduce_policy = RAJA::cuda_reduce;
#elif defined(ASCENT_HIP_ENABLED)
  using reduce_policy = RAJA::hip_reduce;
#else
  using reduce_policy = RAJA::omp_reduce;
#endif
  using atomic_policy = RAJA::omp_atomic;
  static std::string memory_space;
};
#endif

//---------------------------------------------------------------------------//
struct SerialExec
{
  using for_policy = RAJA::seq_exec;
#if defined(ASCENT_CUDA_ENABLED)
  // the cuda/hip policy for reductions can be used
  // by other backends, and this should suppress
  // erroneous host device warnings
  using reduce_policy = RAJA::cuda_reduce;
#elif  defined(ASCENT_HIP_ENABLED)
  using reduce_policy = RAJA::hip_reduce;
#else
  using reduce_policy = RAJA::seq_reduce;
#endif
  using atomic_policy = RAJA::seq_atomic;
  static std::string memory_space;
};

//---------------------------------------------------------------------------//
#if defined(ASCENT_CUDA_ENABLED)
using for_policy    = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
using reduce_policy = RAJA::cuda_reduce;
using atomic_policy = RAJA::cuda_atomic;
#elif defined(ASCENT_HIP_ENABLED)
using for_policy    = RAJA::hip_exec<HIP_BLOCK_SIZE>;
using reduce_policy = RAJA::hip_reduce;
using atomic_policy = RAJA::hip_atomic;
#elif defined(ASCENT_OPENMP_ENABLED)
using for_policy    = RAJA::omp_parallel_for_exec;
using reduce_policy = RAJA::omp_reduce;
using atomic_policy = RAJA::omp_atomic;
#else
using for_policy    = RAJA::seq_exec;
using reduce_policy = RAJA::seq_reduce;
using atomic_policy = RAJA::seq_atomic;
#endif

//---------------------------------------------------------------------------//
//
// CPU only policies need when using classes
// that cannot be called on a GPU, e.g. MFEM
//
#if defined(ASCENT_OPENMP_ENABLED)
using for_cpu_policy    = RAJA::omp_parallel_for_exec;
using reduce_cpu_policy = RAJA::omp_reduce;
using atomic_cpu_policy = RAJA::omp_atomic;
#else
using for_cpu_policy    = RAJA::seq_exec;
using reduce_cpu_policy = RAJA::seq_reduce;
using atomic_cpu_policy = RAJA::seq_atomic;
#endif

//---------------------------------------------------------------------------//
#else
//---------------------------------------------------------------------------//
// policies for when raja is OFF
//---------------------------------------------------------------------------//
struct EmptyPolicy
{};
//---------------------------------------------------------------------------//
struct SerialExec
{
  using for_policy    = EmptyPolicy;
  using reduce_policy = EmptyPolicy;
  using atomic_policy = EmptyPolicy;
  static std::string memory_space;
};

//---------------------------------------------------------------------------//
using for_policy    = EmptyPolicy;
using reduce_policy = EmptyPolicy;
using atomic_policy = EmptyPolicy;

//---------------------------------------------------------------------------//
using for_cpu_policy    = EmptyPolicy;
using reduce_cpu_policy = EmptyPolicy;
using atomic_cpu_policy = EmptyPolicy;
#endif

//---------------------------------------------------------------------------//
// Lambda decorators
//---------------------------------------------------------------------------//

#if defined(__CUDACC__) && !defined(DEBUG_CPU_ONLY)
//---------------------------------------------------------------------------//
// CUDA decorators
//---------------------------------------------------------------------------//
#define ASCENT_EXEC inline __host__ __device__
// Note: there is a performance hit for doing both host and device
// the cuda compiler calls this on then host as a std::function call for each i
// in the for loop, and that basically works out to a virtual function
// call. Thus for small loops, the know overhead is about 3x
#define ASCENT_LAMBDA __device__ __host__

#elif defined(ASCENT_HIP_ENABLED)
//---------------------------------------------------------------------------//
// HIP decorators
//---------------------------------------------------------------------------//
    #error hip support here
#else
//---------------------------------------------------------------------------//
// Non-device decorators
//---------------------------------------------------------------------------//
#define ASCENT_EXEC   inline
#define ASCENT_LAMBDA

#endif

#define ASCENT_CPU_LAMBDA

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
  #error HIP support here
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    std::cerr<<"HIP error reported at: "<<file<<":"<<line;
    std::cerr<<" : "<<cudaGetErrorString(err)<<"\n";
    //exit( -1 );
  }
}
#define ASCENT_DEVICE_ERROR_CHECK() hip_error_check(__FILE__,__LINE__);
#else
//---------------------------------------------------------------------------//
// non-device error check (noop)
//---------------------------------------------------------------------------//
#define ASCENT_DEVICE_ERROR_CHECK()
#endif


} // namespace ascent
#endif
