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
  inline Error(const std::string message) : m_message(message) {}
  inline const std::string & GetMessage() const { return this->m_message; }
  inline const char * what() const noexcept override { return m_message.c_str(); }

};

} // namespace vtkh
#endif
