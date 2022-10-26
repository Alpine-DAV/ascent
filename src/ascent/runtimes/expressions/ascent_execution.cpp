#include "ascent_execution.hpp"
#include <ascent_config.h>
#include <ascent_logging.hpp>

namespace ascent
{
// set the default execution env
#if defined(ASCENT_CUDA_ENABLED)
std::string ExecutionManager::m_exec = "cuda";
#elif defined(ASCENT_HIP_ENABLED)
std::string ExecutionManager::m_exec = "hip";
#elif defined(ASCENT_OPENMP_ENABLED)
std::string ExecutionManager::m_exec = "openmp";
#else
std::string ExecutionManager::m_exec = "serial";
#endif

conduit::Node
ExecutionManager::info()
{
  conduit::Node res;
  res["backends"].append() = "serial";
#if defined(ASCENT_OPENMP_ENABLED)
  res["backends"].append() = "openmp";
#endif
#if defined(ASCENT_CUDA_ENABLED)
  res["backends"].append() = "cuda";
#endif
#if defined(ASCENT_HIP_ENABLED)
  res["backends"].append() = "hip";
#endif
  return res;
}

std::string ExecutionManager::preferred_cpu_device()
{
  std::string res = "serial";

#if defined(ASCENT_OPENMP_ENABLED)
  res = "openmp";
#endif
  return res;
}

std::string ExecutionManager::preferred_gpu_device()
{
  std::string res = "none";

#if defined(ASCENT_CUDA_ENABLED)
  res = "cuda";
#elif defined(ASCENT_HIP_ENABLED)
  res = "hip";
#endif
  return res;
}

void
ExecutionManager::execution(const std::string exec)
{
  if(exec != "cuda" && exec != "openmp" && exec != "serial")
  {
    ASCENT_ERROR("Unknown execution backend '"<<exec<<"')");
  }

#if not defined(ASCENT_CUDA_ENABLED)
  if(exec == "cuda")
  {
    ASCENT_ERROR("Cuda backend support not built");
  }
#endif

#if not defined(ASCENT_HIP_ENABLED)
  if(exec == "hip")
  {
    ASCENT_ERROR("Hip backend support not built");
  }
#endif

#if not defined(ASCENT_OPENMP_ENABLED)
  if(exec == "openmp")
  {
    ASCENT_ERROR("OpenMP backend support not built");
  }
#endif
 
  m_exec = exec;
}

std::string ExecutionManager::execution()
{
  return m_exec;
}

} // namespace ascent
