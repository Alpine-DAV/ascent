#ifndef ASCENT_EXECUTION_HPP
#define ASCENT_EXECUTION_HPP

#include <conduit.hpp>

namespace ascent
{

class ExecutionManager
{
public:
  static conduit::Node info();
  static void execution(const std::string exec);
  static std::string execution();
private:
  static std::string m_exec;
};


} // namespace ascent
#endif
