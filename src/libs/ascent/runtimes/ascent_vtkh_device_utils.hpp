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


using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class ASCENT_API VTKHDeviceAdapter
{
public:

//---------------------------------------------------------------------------//
// Lambda decorators
//---------------------------------------------------------------------------//

#if defined(__CUDACC__) && !defined(DEBUG_CPU_ONLY)
//---------------------------------------------------------------------------//
// CUDA decorators
//---------------------------------------------------------------------------//
static const std::string memory_space = "device";
#define ASCENT_EXEC inline __host__ __device__
// Note: there is a performance hit for doing both host and device
// the cuda compiler calls this on then host as a std::function call for each i
// in the for loop, and that basically works out to a virtual function
// call. Thus for small loops, the know overhead is about 3x
#define ASCENT_LAMBDA __device__ __host__
//#if defined(ASCENT_RAJA_ENABLED)
//#define BLOCK_SIZE 128
//using for_policy = RAJA::cuda_exec<BLOCK_SIZE>;
//using reduce_policy = RAJA::cuda_reduce;
//using atomic_policy = RAJA::cuda_atomic;
//#else
//using for_policy    = EmptyPolicy;
//using reduce_policy = EmptyPolicy;
//using atomic_policy = EmptyPolicy;
//#endif

#elif defined(ASCENT_HIP_ENABLED) // && ?
//---------------------------------------------------------------------------//
// HIP decorators
//---------------------------------------------------------------------------//
static const std::string memory_space = "device";
#define ASCENT_EXEC inline __host__ __device__
#define ASCENT_LAMBDA __device__ __host__
//#if defined(ASCENT_RAJA_ENABLED)
//#define BLOCK_SIZE 256
//using for_policy = RAJA::hip_exec<BLOCK_SIZE>;
//using reduce_policy = RAJA::hip_reduce;
//using atomic_policy = RAJA::hip_atomic;
//#else
//using for_policy    = EmptyPolicy;
//using reduce_policy = EmptyPolicy;
//using atomic_policy = EmptyPolicy;
//#endif

#else
//---------------------------------------------------------------------------//
// Non-device decorators
//---------------------------------------------------------------------------//
static const std::string memory_space = "host";
#define ASCENT_EXEC   inline
#define ASCENT_LAMBDA
//#if defined(ASCENT_RAJA_ENABLED)
//using for_policy = RAJA::seq_exec;
//using reduce_policy = RAJA::seq_reduce;
//using atomic_policy = RAJA::seq_atomic;
//#else
//using for_policy    = EmptyPolicy;
//using reduce_policy = EmptyPolicy;
//using atomic_policy = EmptyPolicy;
//#endif
#endif

// Definition of the cast function with __host__ __device__
ASCENT_EXEC static void castUint64ToFloat64(const uint64_t* input, double* output, int size);


};
//-----------------------------------------------------------------------------
// -- end VTKHDeviceAdapter --
//-----------------------------------------------------------------------------

};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
#endif
