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
#include "ascent_expression_eval.hpp"
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
              bool required,
              bool supports_expressions)
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
    if(params[path].dtype().is_string() && supports_expressions)
    {
      // check to see if this is a valid expression
      is_expr = is_valid_expression(params[path].as_string());
    }

    if(!params[path].dtype().is_number() && !is_expr)
    {
      if(supports_expressions)
      {
        std::string msg = "Numeric parameter '" + path +
                          " : " + params[path].to_yaml()
                             + "'  is not numeric and is not a valid expression";
        info["errors"].append() = msg;
      }
      else
      {
        std::string msg = "Numeric parameter '" + path +
                          " : " + params[path].to_yaml()
                             + "'  is not numeric and does not support expressions";
      }
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

template<typename T>
T conduit_cast(const conduit::Node &node);

template<>
int conduit_cast<int>(const conduit::Node &node)
{
  return node.to_int32();
}

template<>
double conduit_cast<double>(const conduit::Node &node)
{
  return node.to_float64();
}

template<>
float conduit_cast<float>(const conduit::Node &node)
{
  return node.to_float32();
}

template<typename T>
T get_value(const conduit::Node &node, DataObject *dataset)
{
  T value = 0;
  if(node.dtype().is_empty())
  {
    // don't silently return a value from an empty node
    ASCENT_ERROR("Cannot get value from and empty node");
  }

  // check to see if this is an expression
  if(node.dtype().is_string())
  {
    if(dataset == nullptr)
    {
      ASCENT_ERROR("Numeric parameter is an expression(string)"
                   <<" but we can not evaluate the expression."
                   <<" This is usaully for a parameter that is "
                   <<"not meant to have an expression. expression '"
                   <<node.to_string()<<"'");

    }
    // TODO: we want to zero copy this
    conduit::Node * bp_dset = dataset->as_low_order_bp().get();
    expressions::ExpressionEval eval(bp_dset);
    std::string expr = node.as_string();
    conduit::Node res = eval.evaluate(expr);

    if(!res.has_path("value"))
    {
      ASCENT_ERROR("expression '"<<expr
                   <<"': failed to extract a value from the result."
                   <<" "<<res.to_yaml());
    }

    if(res["value"].dtype().number_of_elements() != 1)
    {
      ASCENT_ERROR("expression '"<<expr
                   <<"' resulted in multiple values."
                   <<" Expected scalar. "<<res.to_yaml());
    }
    value = res["value"].to_float64();
  }
  else
  {
    value = conduit_cast<T>(node);
  }
  return value;
}

double get_float64(const conduit::Node &node, DataObject *dataset)
{
  return get_value<double>(node, dataset);
}

float get_float32(const conduit::Node &node, DataObject *dataset)
{
  return get_value<float>(node, dataset);
}

int get_int32(const conduit::Node &node, DataObject *dataset)
{
  return get_value<int>(node, dataset);
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





