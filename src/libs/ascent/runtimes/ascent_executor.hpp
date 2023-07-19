//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_executor.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_EXECUTOR_HPP
#define ASCENT_EXECUTOR_HPP

#include <ascent.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

static std::map<std::string, bool (*)(void)> m_callback_map;

class ASCENT_API Executor
{
public:

  void static register_callback(const std::string &callback_name,
                                bool (*callback_function)(void));
  void static execute(const std::string &command,
                      const std::string &command_type);
  void static execute_callback(const std::string &callback_name);
  void static execute_shell_command(const std::string &command);

};

//-----------------------------------------------------------------------------
};
#endif
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
