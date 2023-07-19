//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_executor.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_executor.hpp"

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

void Executor::register_callback(const std::string &callback_name,
                                bool (*callback_function)(void))
{
  m_callback_map.insert(std::make_pair(callback_name, callback_function));
}

void Executor::execute(const std::string &command,
                       const std::string &command_type)
{
  if (command_type == "callback")
  {
    execute_callback(command);
  }
  else
  {
    execute_shell_command(command);
  }
}

void Executor::execute_callback(const std::string &callback_name)
{
  auto callback_pair = m_callback_map.find(callback_name);
  auto callback_function = callback_pair->second;
  callback_function();
}

void Executor::execute_shell_command(const std::string &shell_command)
{
  system(shell_command.c_str());
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
