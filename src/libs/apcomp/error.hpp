#ifndef APCOMP_ERROR_HPP
#define APCOMP_ERROR_HPP

#include <apcomp/apcomp_exports.h>
#include <exception>
#include <string>

namespace apcomp {

class APCOMP_API Error : public std::exception
{
private:
  std::string m_message;
  Error() {}
public:
  Error(const std::string message) : m_message(message) {}
  const std::string & GetMessage() const { return this->m_message; }
  const char * what() const noexcept override { return m_message.c_str(); }

};

} // namespace vtkh
#endif
