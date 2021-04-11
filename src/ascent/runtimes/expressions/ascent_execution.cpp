#include "ascent_execution.hpp"
#include <ascent_config.h>
#include <ascent_logging.hpp>

namespace ascent
{
// set the default exection env
#ifdef ASCENT_USE_CUDA
std::string ExecutionManager::m_exec = "cuda";
#elif defined(ASCENT_USE_OPENMP)
std::string ExecutionManager::m_exec = "openmp";
#else
std::string ExecutionManager::m_exec = "serial";
#endif

conduit::Node
ExecutionManager::info()
{
  conduit::Node res;
  res["backends"].append() = "serial";
#ifdef ASCENT_USE_CUDA
  res["backends"].append() = "cuda";
#endif
#if defined(ASCENT_USE_OPENMP)
  res["backends"].append() = "openmp";
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

#ifndef ASCENT_USE_CUDA
  if(exec == "cuda")
  {
    ASCENT_ERROR("Cuda backend support not built");
  }
#endif

#if not defined(ASCENT_USE_OPENMP)
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
