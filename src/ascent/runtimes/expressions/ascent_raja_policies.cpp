#include "ascent_raja_policies.hpp"
#include <ascent_logging.hpp>

namespace ascent
{

#ifdef ASCENT_USE_CUDA
std::string CudaExec::memory_space = "device";
#endif

#if defined(ASCENT_USE_OPENMP)
std::string OpenMPExec::memory_space = "host";
#endif

std::string SerialExec::memory_space = "host";

} // namespace ascent
