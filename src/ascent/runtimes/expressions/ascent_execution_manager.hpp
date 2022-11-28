#ifndef ASCENT_EXECUTION_HPP
#define ASCENT_EXECUTION_HPP

#include <conduit.hpp>
#include <ascent_exports.h>

namespace ascent
{

//-----------------------------------------------------------------------------
class ASCENT_API ExecutionManager
{
public:
  static conduit::Node info();
  static void          set_execution_policy(const std::string &exec);
  static std::string   execution_policy();

  // return the preferred cpu election device
  // i.e., openmp if supported and serial if not
  static std::string preferred_cpu_policy();

  // return the preferred gpuu election device
  // i.e., none, cuda, or hip
  static std::string preferred_gpu_policy();
private:
  static std::string m_exec;
};


} // namespace ascent
#endif
