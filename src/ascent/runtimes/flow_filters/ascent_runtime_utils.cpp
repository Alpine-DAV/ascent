//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_utils.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_utils.hpp"
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_metadata.hpp>

#include <algorithm>

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

std::string output_dir(const std::string file_name)
{
  std::string output_path;

  std::string file, base_path;
  conduit::utils::rsplit_file_path(file_name, file, base_path);
  if(base_path == "")
  {
    std::string dir = default_dir();
    output_path = conduit::utils::join_file_path(dir, file);
  }
  else
  {
    output_path = file_name;
  }
  return output_path;
}

std::string default_dir()
{
  if(Metadata::n_metadata.has_path("default_dir"))
  {
    return Metadata::n_metadata["default_dir"].as_string();
  }
  else return ".";
}

std::string filter_to_path(const std::string filter_name)
{
  std::string res;
  std::vector<std::string> path = split(filter_name, '_');
  for(size_t i = 0; i < path.size(); ++i)
  {
    res += path[i];
    if(i != path.size() - 1)
    {
      res += "/";
    }
  }
  return res;
}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





