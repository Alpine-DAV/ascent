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
#include <regex>
#include <conduit.hpp>


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

template<typename T>
std::string expand_format_value(const std::string path_string, const T value);

template<typename T>
std::string expand_generic_variable(const std::string& path_string, const std::regex& pattern, const T value);

// keeps track of static counters for a given key, i.e., name
// c style print formatting is supported as well as ascent path 
// formatting notation. For example, "file_%04d_{family:%03d}"
// would expand to "file_0001_001", if the counter for that key is 1.
// If no formatting is present, the count is appended to the name.
int get_family_value(const std::string& path_string, int family_value);

// searches for previously defined keywords in a string and fills 
// in the string with their values. Current supported special variables
// are cycle, family, and time. 
std::string expand_path_special_variables(const std::string& path_string, const conduit::Node &meta, int counter = 0);

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


