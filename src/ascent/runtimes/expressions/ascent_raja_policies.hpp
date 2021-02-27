#ifndef ASCENT_RAJA_POLICIECS_HPP
#define ASCENT_RAJA_POLICIECS_HPP

#include <ascent_config.h>
#include <RAJA/RAJA.hpp>

namespace ascent
{

#ifdef ASCENT_USE_CUDA
#define BLOCK_SIZE 128
using for_policy = RAJA::cuda_exec<BLOCK_SIZE>;
using reduce_policy = RAJA::cuda_reduce;
using atomic_policy = RAJA::cuda_atomic;
#elif ASCENT_USE_OPENMP
using for_policy = RAJA::omp_parallel_for_exec;
using reduce_policy = RAJA::omp_reduce;
using atomic_policy = RAJA::omp_atomic;
#else
using for_policy = RAJA::seq_exec;
using reduce_policy = RAJA::seq_reduce;
using atomic_policy = RAJA::seq_atomic;
#endif

//
// CPU only policies need when using classes
// that cannot be called on a GPU, e.g. MFEM
//
#ifdef ASCENT_USE_OPENMP
using for_cpu_policy = RAJA::omp_parallel_for_exec;
using reduce_cpu_policy = RAJA::omp_reduce;
using atomic_cpu_policy = RAJA::omp_atomic;
#else
using for_cpu_policy = RAJA::seq_exec;
using reduce_cpu_policy = RAJA::seq_reduce;
using atomic_cpu_policy = RAJA::seq_atomic;
#endif

// -------------------- Lambda decorators ----------------------
#if defined(__CUDACC__) && !defined(DEBUG_CPU_ONLY)

#define ASCENT_USE_CUDA
#define ASCENT_EXEC inline __host__ __device__
#define ASCENT_LAMBDA __device__

#else

#define ASCENT_EXEC inline
#define ASCENT_LAMBDA

#endif

#define ASCENT_CPU_LAMBDA

// -------------------- Error Checking --------------------------
#ifdef ASCENT_USE_CUDA
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
#define ASCENT_ERROR_CHECK() cuda_error_check(__FILE__,__LINE__);
#else
#define ASCENT_ERROR_CHECK()
#endif


} // namespace ascent
#endif
