#ifndef VTK_H_ERROR_HPP
#define VTK_H_ERROR_HPP

#include <vtkh/vtkh_exports.h>
#include <exception>
#include <sstream>
#include <string>

namespace vtkh {

class VTKH_API Error : public std::exception
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
