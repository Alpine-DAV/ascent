#ifndef ASCENT_RAJA_POLICIECS_HPP
#define ASCENT_RAJA_POLICIECS_HPP

#include <ascent_config.h>
#include <conduit.hpp>
#include <RAJA/RAJA.hpp>

namespace ascent
{

#ifdef ASCENT_USE_CUDA
#define CUDA_BLOCK_SIZE 128
struct CudaExec
{
  using for_policy = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
  using reduce_policy = RAJA::cuda_reduce;
  using atomic_policy = RAJA::cuda_atomic;
  static std::string memory_space;
};
#endif

#if defined(ASCENT_USE_OPENMP)
struct OpenMPExec
{
  using for_policy = RAJA::omp_parallel_for_exec;
#ifdef ASCENT_USE_CUDA
  // the cuda policy for reductions can be used
  // by other backends, and this should suppress
  // erroneous host device warnings
  using reduce_policy = RAJA::cuda_reduce;
#else
  using reduce_policy = RAJA::omp_reduce;
#endif
  using atomic_policy = RAJA::omp_atomic;
  static std::string memory_space;
};
#endif

struct SerialExec
{
  using for_policy = RAJA::seq_exec;
#ifdef ASCENT_USE_CUDA
  // the cuda policy for reductions can be used
  // by other backends, and this should suppress
  // erroneous host device warnings
  using reduce_policy = RAJA::cuda_reduce;
#else
  using reduce_policy = RAJA::seq_reduce;
#endif
  using atomic_policy = RAJA::seq_atomic;
  static std::string memory_space;
};

#ifdef ASCENT_USE_CUDA
#define BLOCK_SIZE 128
using for_policy = RAJA::cuda_exec<BLOCK_SIZE>;
using reduce_policy = RAJA::cuda_reduce;
using atomic_policy = RAJA::cuda_atomic;
#elif defined(ASCENT_USE_OPENMP)
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

#define ASCENT_EXEC inline __host__ __device__
// Note: there is a performance hit for doing both host and device
// the cuda compiler calls this on then host as a std::function call for each i
// in the for loop, and that basically works out to a virtual function
// call. Thus for small loops, the know overhead is about 3x
#define ASCENT_LAMBDA __device__ __host__

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
