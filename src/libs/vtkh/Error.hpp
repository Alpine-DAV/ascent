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
  Error();
public:
  Error(const std::string message);
  ~Error() override;
  const std::string & GetMessage() const;
  const char * what() const noexcept override;

};

} // namespace vtkh
#endif
