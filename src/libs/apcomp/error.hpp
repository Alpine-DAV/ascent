//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef APCOMP_ERROR_HPP
#define APCOMP_ERROR_HPP

#include <apcomp/apcomp_config.h>
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
