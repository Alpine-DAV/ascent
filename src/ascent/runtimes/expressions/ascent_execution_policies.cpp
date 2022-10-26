#include "ascent_execution_policies.hpp"
#include <ascent_logging.hpp>

namespace ascent
{

#if defined(ASCENT_CUDA_ENABLED)
std::string CudaExec::memory_space = "device";
#endif

#if defined(ASCENT_HIP_ENABLED)
std::string HipExec::memory_space = "device";
#endif

#if defined(ASCENT_OPENMP_ENABLED)
std::string OpenMPExec::memory_space = "host";
#endif

std::string SerialExec::memory_space = "host";

} // namespace ascent
