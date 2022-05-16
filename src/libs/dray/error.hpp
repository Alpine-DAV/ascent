// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ERROR_HPP
#define DRAY_ERROR_HPP

#include <exception>
#include <string>
#include <sstream>

namespace dray
{

class DRayError : public std::exception
{
  private:
  std::string m_message;
  DRayError ()
  {
  }

  public:
  DRayError (const std::string message, const std::string file, int line)
  {
    std::stringstream msg;
    msg<<file<<" ("<<line<<"): "<<message<<"\n";
    m_message = msg.str();
  }
  const std::string &GetMessage () const
  {
    return this->m_message;
  }
  const char *what () const noexcept override
  {
    return m_message.c_str ();
  }
};

#define DRAY_ERROR( msg )                       \
{                                               \
    std::ostringstream oss_error;               \
    oss_error << msg;                           \
    throw DRayError(oss_error.str(),            \
                    std::string(__FILE__),      \
                     __LINE__);                 \
}                                               \

} // namespace dray
#endif
