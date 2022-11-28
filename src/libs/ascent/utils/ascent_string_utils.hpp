//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_string_utils.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_STRING_UTILS_HPP
#define ASCENT_STRING_UTILS_HPP

#include <string>
#include <vector>


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{
// keeps track of static counters for a given key, i.e., name
// c style print formatting is supported. For example, "file_%04d"
// would expand to "file_0001", if the counter for that key is 1.
// If no formatting is present, the count is appended to the name.
std::string expand_family_name(const std::string name, int counter = 0);

std::vector<std::string> split(const std::string &s, char delim = ' ');

std::string timestamp();

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


