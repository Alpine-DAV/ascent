#include <vtkh/Error.hpp>

namespace vtkh
{

Error::Error() = default;

Error::Error(const std::string message)
  : m_message(message)
{
}

Error::~Error() = default;

const std::string & Error::GetMessage() const
{
  return this->m_message;
}

const char * Error::what() const noexcept
{
  return m_message.c_str();
}

}

