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
/// file: ascent_string_utils.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_actions_utils.hpp"
#include <map>
#include <sstream>
#include <stdio.h>
#include <regex>
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

namespace detail
{
void parse_expression(const std::string &expression,
                      std::vector<std::string> &fields)
{
  std::regex e ("field\\('(.*?)'\\)");
  std::smatch m;

  std::vector<std::string> matches;
  std::string s = expression;
  while (std::regex_search (s,m,e))
  {
    int count = 0;
    for (auto x:m)
    {
      // we  want the second submatch that
      // matches the regex  inside the single
      // quotes
      if(count == 1)
      {
        fields.push_back(x);
      }
      count++;
    }
    std::cout << std::endl;
    s = m.suffix().str();
  }
}

void filter_fields(const conduit::Node &node,
                   std::vector<std::string> &fields,
                   conduit::Node &info)
{
  const int num_children = node.number_of_children();
  const std::vector<std::string> names = node.child_names();
  for(int i = 0; i < num_children; ++i)
  {
    const conduit::Node &child = node.child(i);
    bool is_leaf = child.number_of_children() == 0;
    if(is_leaf)
    {
      if(names[i] == "field")
      {
        fields.push_back(child.as_string());
      }
      if(names[i] == "expression")
      {
        parse_expression(child.as_string(), fields);
      }
      // special detection for filters that use
      // all fields by default
      if(is_leaf && (names[i]  == "type") )
      {
        const std::string type = child.as_string();
        if(type == "relay" ||
           type == "project_2d" ||
           type == "dray_project_2d")
        {
          if(!node.has_path("params/fields"))
          {
            conduit::Node &error = info.append();
            error["filter"] = type;
            error["message"] = "filter does not specify what fields "
                               "to use.";
          }
        }

      } // is type
    } // is leaf

    if(!is_leaf)
    {
      if(!child.dtype().is_list())
      {
        filter_fields(child, fields, info);
      }
      else
      {
        // this is a list
        if(names[i] == "fields")
        {
          const int num_entries = child.number_of_children();
          for(int e = 0; e < num_entries;  e++)
          {
            const conduit::Node &item = child.child(e);
            if(item.dtype().is_string())
            {
              fields.push_back(item.as_string());
            }
          } // for list  entries
        } // is  field list
      } //  list processing
    } // inner node
  } // for children
}

} // namespace detail

bool field_list(const conduit::Node &actions,
                              std::vector<std::string> &fields,
                              conduit::Node &info)
{
  info.reset();
  fields.clear();
  detail::filter_fields(actions, fields, info);
  for(int i = 0; i < fields.size(); ++i) std::cout<<fields[i]<<"\n";
  if(info.number_of_children() != 0)
  {
    info.print();
  }
  return info.number_of_children() == 0;
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



