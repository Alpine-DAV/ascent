#ifndef ASCENT_EXECUTION_POLICIES_HPP
#define ASCENT_EXECUTION_POLICIES_HPP

#include <ascent_config.h>
#include <limits>
#include <conduit.hpp>


#if defined(ASCENT_RAJA_ENABLED)
#include <RAJA/RAJA.hpp>
#endif

namespace ascent
{
    
    
//---------------------------------------------------------------------------//
/*
//---------------------------------------------------------------------------//
Ascent Device Execution Policies and Interfaces.


-----------------------
Execution Policies 
-----------------------

Provides structs with typed policies used to select host or device execution
in various dispatch methods. 

struct ExecPolicy
{
    using for_policy     = ... // determines for loop execution
    using reduce_policy  = ... // determines reduction execution
    using atomic_policy  = ... // determines atomic operations execution
}

When RAJA is enabled, RAJA policy types are selected.

When RAJA is disabled, stub policies are provided and serial execution is
only option.


-----------------------
Execution Interfaces 
-----------------------

Provides forall, reduction, and atomic interfaces:

* forall
* ReduceSum
* ReduceMin
* ReduceMinLoc
* ReduceMax
* ReduceMaxLoc
* atomic_add
* atomic_min
* atomic_max

When RAJA is enabled, RAJA execution is used. 

When RAJA is disabled, serial implementations are substituted.


forall usage example:

  ascent::forall<Exec>(0, size, [=] ASCENT_LAMBDA(index_t i)
  {
    data[i] = ...;
  });

//---------------------------------------------------------------------------//
*/
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Index type
//---------------------------------------------------------------------------//
using index_t = conduit::index_t;


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

#elif defined(ASCENT_HIP_ENABLED) // && ?
//---------------------------------------------------------------------------//
// HIP decorators
//---------------------------------------------------------------------------//
#define ASCENT_EXEC inline __host__ __device__
#define ASCENT_LAMBDA __device__ __host__

#else
//---------------------------------------------------------------------------//
// Non-device decorators
//---------------------------------------------------------------------------//
#define ASCENT_EXEC   inline
#define ASCENT_LAMBDA

#endif

#define ASCENT_CPU_LAMBDA


//---------------------------------------------------------------------------//
#if defined(ASCENT_RAJA_ENABLED)
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// RAJA_ON policies for when raja is on
//---------------------------------------------------------------------------//
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
// RAJA Exec Interfaces
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// forall
//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename Kernel>
inline void forall(const index_t& begin,
                   const index_t& end,
                   Kernel&& kernel) noexcept
{
    RAJA::forall<ExecPolicy>(RAJA::RangeSegment(begin, end),
                             std::forward<Kernel>(kernel));
}

//---------------------------------------------------------------------------//
// Reductions
//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
using ReduceSum    = RAJA::ReduceSum<ExecPolicy,T>;

template <typename ExecPolicy, typename T>
using ReduceMin    = RAJA::ReduceMin<ExecPolicy,T>;
template <typename ExecPolicy, typename T>
using ReduceMinLoc    = RAJA::ReduceMinLoc<ExecPolicy,T>;

template <typename ExecPolicy, typename T>
using ReduceMax    = RAJA::ReduceMax<ExecPolicy,T>;
template <typename ExecPolicy, typename T>
using ReduceMaxLoc = RAJA::ReduceMaxLoc<ExecPolicy,T>;

//---------------------------------------------------------------------------//
// Atomics
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
ASCENT_EXEC T atomic_add(T volatile *acc, T value)
{
    return RAJA::atomicAdd(ExecPolicy{}, acc, value);
}

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
ASCENT_EXEC T atomic_min(T volatile *acc, T value)
{
    return RAJA::atomicMin(ExecPolicy{}, acc, value);
}

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
ASCENT_EXEC T atomic_max(T volatile *acc, T value)
{
    return RAJA::atomicMax(ExecPolicy{}, acc, value);
}


//---------------------------------------------------------------------------//
#else
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
// RAJA_OFF policies for when raja is OFF
//---------------------------------------------------------------------------//
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

//---------------------------------------------------------------------------//
// Exec interfaces for when RAJA is disabled
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
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


//---------------------------------------------------------------------------//
// Reductions
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// -------
// For the curious: 
// -------
// the const crimes we commit here are in the name of [=] capture
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
class ReduceSum
{
public:
    //---------------------------------------------------------------------
    ReduceSum()
    : m_value(0),
      m_value_ptr(&m_value)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceSum(T v_start)
    : m_value(v_start),
      m_value_ptr(&m_value)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceSum(const ReduceSum &v)
    : m_value(v.m_value), // will be unused in copies
      m_value_ptr(v.m_value_ptr) // this is where the magic happens
    {
        // empty
    }

    //---------------------------------------------------------------------
    void sum(const T value) const
    {
        m_value_ptr[0] += value;
    }

    //---------------------------------------------------------------------
    T get() const
    {
        return m_value;
    }

private:
    T  m_value;
    T* m_value_ptr;
};


//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
class ReduceMin
{
public:
    
    //---------------------------------------------------------------------
    ReduceMin()
    : m_value(std::numeric_limits<T>::max()),
      m_value_ptr(&m_value)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMin(T v_start)
    : m_value(v_start),
      m_value_ptr(&m_value)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMin(const ReduceMin &v)
    : m_value(v.m_value), // will be unused in copies
      m_value_ptr(v.m_value_ptr) // this is where the magic happens
    {
        // empty
    }
    
    //---------------------------------------------------------------------
    void min(const T value) const
    {
        if (value < m_value_ptr[0])
        {
            m_value_ptr[0]=value;
        }
    }

    //---------------------------------------------------------------------
    T get() const
    {
        return m_value_ptr[0];
    }

private:
    T  m_value;
    T *m_value_ptr;
};

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
class ReduceMinLoc
{
public:

    //---------------------------------------------------------------------
    ReduceMinLoc()
    : m_value(std::numeric_limits<T>::max()),
      m_value_ptr(&m_value),
      m_index(-1),
      m_index_ptr(&m_index)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMinLoc(T v_start, index_t i_start)
    : m_value(v_start),
      m_value_ptr(&m_value),
      m_index(i_start),
      m_index_ptr(&m_index)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMinLoc(const ReduceMinLoc &v)
    : m_value(v.m_value), // will be unused in copies
      m_value_ptr(v.m_value_ptr), // this is where the magic happens
      m_index(v.m_index), // will be unused in copies
      m_index_ptr(v.m_index_ptr) // this is where the magic happens
    {
        // empty
    }

    //---------------------------------------------------------------------
    inline void minloc(const T v, index_t i) const
    {
        if(v < m_value_ptr[0])
        {
            m_value_ptr[0]=v;
            m_index_ptr[0]=i;
        }
    };

    //---------------------------------------------------------------------
    inline T get() const
    {
        return m_value_ptr[0];
    }

    //---------------------------------------------------------------------
    inline index_t getLoc() const
    {
        return m_index_ptr[0];
    }

private:
    T         m_value;
    T       *m_value_ptr;
    index_t   m_index;
    index_t  *m_index_ptr;
};

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
class ReduceMax
{
public:
    //---------------------------------------------------------------------
    ReduceMax()
    : m_value(std::numeric_limits<T>::lowest()),
      m_value_ptr(&m_value)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMax(T v_start)
    : m_value(v_start),
      m_value_ptr(&m_value)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMax(const ReduceMax &v)
    : m_value(v.m_value), // will be unused in copies
      m_value_ptr(v.m_value_ptr) // this is where the magic happens
    {
        // empty
    }

    //---------------------------------------------------------------------
    // the const crimes we commit here are in the name of [=] capture
    void max(const T value) const
    {
        if (value >  m_value_ptr[0])
        {
            m_value_ptr[0]=value;
        }
    }
    
    //---------------------------------------------------------------------
    T get() const
    {
        return  m_value_ptr[0];
    }

private:
    T  m_value;
    T *m_value_ptr; 
};

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
class ReduceMaxLoc
{
public:

    //---------------------------------------------------------------------
    ReduceMaxLoc()
    : m_value(std::numeric_limits<T>::lowest()),
      m_value_ptr(&m_value),
      m_index(-1),
      m_index_ptr(&m_index)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMaxLoc(T v_start, index_t i_start)
    : m_value(v_start),
      m_value_ptr(&m_value),
      m_index(i_start),
      m_index_ptr(&m_index)
    {
        // empty
    }

    //---------------------------------------------------------------------
    ReduceMaxLoc(const ReduceMaxLoc &v)
    : m_value(v.m_value), // will be unused in copies
      m_value_ptr(v.m_value_ptr), // this is where the magic happens
      m_index(v.m_index), // will be unused in copies
      m_index_ptr(v.m_index_ptr) // this is where the magic happens
    {
        // empty
    }

    //---------------------------------------------------------------------
    // the const crimes we commit here are in the name of [=] capture
    inline void maxloc(const T v, index_t i) const
    {
        if(v > m_value_ptr[0])
        {
            m_value_ptr[0] = v;
            m_index_ptr[0] = i;
        }
    };

    //---------------------------------------------------------------------
    inline T get() const
    {
        return m_value_ptr[0];
    }

    //---------------------------------------------------------------------
    inline index_t getLoc() const
    {
        return m_index_ptr[0];
    }

private:
    T        m_value;
    T       *m_value_ptr;
    index_t  m_index;
    index_t *m_index_ptr;
};

//---------------------------------------------------------------------------//
// Atomics
//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
inline T atomic_add(T* acc, T value)
{
    T res = (*acc);
    (*acc)+=value;
    return res;
}

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
inline T atomic_min(T *acc, T value)
{
    T res = (*acc);
    (*acc) = std::min(*acc, value);
    return res;
}

//---------------------------------------------------------------------------//
template <typename ExecPolicy, typename T>
inline T atomic_max(T *acc, T value)
{
    T res = (*acc);
    (*acc) = std::max(*acc, value);
    return res;
}


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


} // namespace ascent
#endif
