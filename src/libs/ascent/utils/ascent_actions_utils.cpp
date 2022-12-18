//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
void parse_binning_var(const std::string &expression,
                       std::set<std::string> &fields)
{

  // find binning variable
  // \\s* = allow for spaces
  std::regex e ("binning\\(\\s*'(.*?)'");
  std::smatch m;

  std::set<std::string> matches;
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
        fields.insert(x);
      }
      count++;
    }
    s = m.suffix().str();
  }
}

void parse_binning_axis(const std::string &expression,
                        std::set<std::string> &fields)
{
  std::regex e ("axis\\(\\s*'(.*?)'");
  std::smatch m;

  std::set<std::string> matches;
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
        // skip coordinate axes
        if(x != "x" && x != "y" && x != "z")
        {
          fields.insert(x);
        }
      }
      count++;
    }
    s = m.suffix().str();
  }
}

void parse_binning(const std::string &expression,
                   std::set<std::string> &fields)
{
  if(expression.find("binning") == std::string::npos)
  {
    return;
  }

  parse_binning_var(expression, fields);
  parse_binning_axis(expression, fields);

}

// some expressions like lineouts take a list of fields
void parse_field_list(const std::string &expression,
                          std::set<std::string> &fields)
{
  // this is a string list of the form fields = ['bananas', 'apples']
  std::regex e ("fields\\s*=\\s*\\[\\s*('[a-zA-Z][a-zA-Z_0-9]*'"
                "\\s*,\\s*)*\\s*'[a-zA-Z][a-zA-Z_0-9]*'\\s*\\]");
  std::smatch m;

  std::set<std::string> matches;
  std::string s = expression;
  std::string flist;
  // just get the entire matched experssion
  if(std::regex_search (s,m,e) == true)
  {
    flist = m.str(0);
  }
  if(flist != "")
  {
    // ok we have a field list now get all the fields
    std::regex inside_single("'([^\\']*)'");
    while(std::regex_search (flist,m,inside_single))
    {
      // the first element in the match is the whole match
      // e.g., 'bananas' and the second is the match inside the
      // quotes = bananas, which is what we want
      auto it = m.begin();
      it++;
      std::string field = *it;
      fields.insert(field);
      // move to the next match
      flist = m.suffix().str();
    }
  }
}

void parse_expression(const std::string &expression,
                      std::set<std::string> &fields)
{
  parse_binning(expression, fields);
  parse_field_list(expression, fields);

  std::regex e ("field\\('(.*?)'\\)");
  std::smatch m;

  std::set<std::string> matches;
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
        fields.insert(x);
      }
      count++;
    }
    s = m.suffix().str();
  }
}

void filter_fields(const conduit::Node &node,
                   std::set<std::string> &fields,
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
      if( names[i] == "field" ||
          names[i] == "field1" || // support for composite vector
          names[i] == "field2" ||
          names[i] == "field3" )
      {
        fields.insert(child.as_string());
      }
      // rover xray
      if(names[i] == "absorption" || names[i] == "emission")
      {
        fields.insert(child.as_string());
      }
      if(names[i] == "expression")
      {
        parse_expression(child.as_string(), fields);
      }
      // special detection for filters that use
      // all fields by default
      if(names[i] == "type")
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
            error["message"] = "The filter does not specify what fields "
                               "to use. In order to use field filtering, "
                               "please consult the Ascent user documentation "
                               "for this filter type to learn how to specify "
                               "specific fields";
          }
        }

      } // is type

      if(names[i] == "actions_file")
      {
        conduit::Node &error = info.append();
        error["message"] = "Field filtering does not support "
                           "scanning actions files specified "
                           "by triggers. Please specifiy the "
                           "trigger actions directly in the "
                           "trigger parameters.";
      } // actions file
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
              fields.insert(item.as_string());
            }
          } // for list  entries
        } // is  field list
      } //  list processing
    } // inner node
  } // for children
}

} // namespace detail

bool field_list(const conduit::Node &actions,
                std::set<std::string> &fields,
                conduit::Node &info)
{
  info.reset();
  fields.clear();
  detail::filter_fields(actions, fields, info);
  return info.number_of_children() == 0;
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



