//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_file_system.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_FILE_SYSTEM_HPP
#define ASCENT_FILE_SYSTEM_HPP

#include <string>
#include <ascent_exports.h>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{


// helper to check if a directory exists
bool ASCENT_API directory_exists(const std::string &path);
// helper to create a directory
bool ASCENT_API create_directory(const std::string &path);

// helper to copy a file to another path
// always overwrites dest_path
bool ASCENT_API copy_file(const std::string &src_path,
                          const std::string &dest_path);

// helper to copy a directory to another path
// always overwrites contents of dest_path
bool ASCENT_API copy_directory(const std::string &src_path,
                               const std::string &dest_path);


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


