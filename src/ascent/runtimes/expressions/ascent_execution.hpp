#ifndef ASCENT_EXECUTION_HPP
#define ASCENT_EXECUTION_HPP

#include <conduit.hpp>
#include <ascent_exports.h>

namespace ascent
{

class ASCENT_API ExecutionManager
{
public:
  static conduit::Node info();
  static void execution(const std::string exec);
  static std::string execution();
  // return the peferred cpu exection device
  // i.e., openmp if supported and serial if not
  static std::string preferred_cpu_device();
private:
  static std::string m_exec;
};


} // namespace ascent
#endif
