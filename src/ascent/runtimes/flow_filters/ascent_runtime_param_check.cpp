//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_param_check.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_param_check.hpp"
#include "expressions/ascent_expressions_ast.hpp"
#include "expressions/ascent_expressions_tokens.hpp"
#include "expressions/ascent_expressions_parser.hpp"
#include <ascent_logging.hpp>

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

// this detects if the syntax is valid, not
// whether the expression will actually work
bool is_valid_expression(const std::string expr)
{
  bool res = true;
  try
  {
    scan_string(expr.c_str());
  }
  catch(const char *msg)
  {
    res = false;
  }
  return res;
}

bool
check_numeric(const std::string path,
              const conduit::Node &params,
              conduit::Node &info,
              bool required)
{
  bool res = true;
  if(!params.has_path(path) && required)
  {
    info["errors"].append() = "Missing required numeric parameter '" + path + "'";
    res = false;
  }

  if(params.has_path(path))
  {

    bool is_expr = false;
    if(params[path].dtype().is_string())
    {
      // check to see if this is a valid expression
      is_expr = is_valid_expression(params[path].as_string());
    }

    if(!params[path].dtype().is_number() && !is_expr)
    {
      std::string msg = "Numeric parameter '" + path +
                        " : " + params[path].to_yaml()
                           + "'  is not numeric and is not a valid expression'";
      info["errors"].append() = msg;
      res = false;
    }
  }
  return res;
}

bool
check_string(const std::string path,
             const conduit::Node &params,
             conduit::Node &info,
             bool required)
{
  bool res = true;
  if(!params.has_path(path) && required)
  {
    info["errors"].append() = "Missing required string parameter '" + path + "'";
    res = false;
  }

  if(params.has_path(path) && !params[path].dtype().is_string())
  {
    std::string msg = "String parameter '" + path + "' is not a string'";
    info["errors"].append() = msg;
    res = false;
  }
  return res;
}

std::string
surprise_check(const std::vector<std::string> &valid_paths,
               const std::vector<std::string> &ignore_paths,
               const conduit::Node &params)
{

  std::stringstream ss;
  std::vector<std::string> paths;
  std::string curr_path = params.path() == "" ? "" :params.path() + "/";
  path_helper(paths, ignore_paths, params, curr_path);
  const int num_paths = static_cast<int>(paths.size());
  const int num_valid_paths = static_cast<int>(valid_paths.size());
  for(int i = 0; i < num_paths; ++i)
  {
    bool found = false;
    for(int f = 0; f < num_valid_paths; ++f)
    {
      if(curr_path + valid_paths[f] == paths[i])
      {
        found = true;
        break;
      }
    }

    if(!found)
    {
      ss<<"Surprise parameter '"<<paths[i]<<"'\n";
    }
  }

  return ss.str();
}

std::string
surprise_check(const std::vector<std::string> &valid_paths,
               const conduit::Node &params)
{

  std::stringstream ss;
  std::vector<std::string> paths;
  path_helper(paths, params);
  const int num_paths = static_cast<int>(paths.size());
  const int num_valid_paths = static_cast<int>(valid_paths.size());
  std::string curr_path = params.path() == "" ? "" :params.path() + "/";
  for(int i = 0; i < num_paths; ++i)
  {
    bool found = false;
    for(int f = 0; f < num_valid_paths; ++f)
    {
      if(curr_path + valid_paths[f] == paths[i])
      {
        found = true;
        break;
      }
    }

    if(!found)
    {
      ss<<"Surprise parameter '"<<paths[i]<<"'\n";
    }
  }

  return ss.str();
}

void
path_helper(std::vector<std::string> &paths, const conduit::Node &node)
{
  const int num_children = static_cast<int>(node.number_of_children());

  if(num_children == 0)
  {
    paths.push_back(node.path());
    return;
  }
  for(int i = 0; i < num_children; ++i)
  {
    const conduit::Node &child = node.child(i);
    path_helper(paths, child);
  }

}

void
path_helper(std::vector<std::string> &paths,
            const std::vector<std::string> &ignore,
            const conduit::Node &params,
            const std::string path_prefix)
{
  const int num_children = static_cast<int>(params.number_of_children());
  const int num_ignore_paths = static_cast<int>(ignore.size());

  for(int i = 0; i < num_children; ++i)
  {
    bool skip = false;
    const conduit::Node &child = params.child(i);
    for(int p = 0; p < num_ignore_paths; ++p)
    {
      const std::string ignore_path = path_prefix + ignore[p];
      if(child.path().compare(0, ignore_path.length(), ignore_path) == 0)
      {
        skip = true;
        break;
      }
    }

    if(!skip)
    {
      path_helper(paths, child);
    }
  }

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





