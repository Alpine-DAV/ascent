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
// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#include <conduit_blueprint_mpi.hpp>
#endif

using namespace conduit;

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

    // to avoid complex parsing, we added a shortcut case
    // action: `declare_fields`
    // fields: ["field1", "field2", .... "fieldN-1"]

    if(child.has_child("action") && child.has_child("fields"))
    {
        if(child["action"].as_string() == "declare_fields")
        {
            const Node &fields_list = child["fields"];
            const int num_entries = fields_list.number_of_children();
            for(int e = 0; e < num_entries;  e++)
            {
                const conduit::Node &item = fields_list.child(e);
                if(item.dtype().is_string())
                {
                    fields.insert(item.as_string());
                }
            } // for list  entries
            // early return, user needs to provide a definitive list
            return;
        }
    }

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

      // ---------
      // special cases for data binning
      // ---------
      // OLD STYLE BINNING:
      // params:
      //   reduction_op: "sum"
      //   var: "braid"
      //   output_field: "binning"
      //   output_type: "bins"
      //   axes:
      //     -
      //       var: "x"
      //       num_bins: 10
      //       min_val: -10.0
      //       max_val: 10.0
      //       clamp: 1
      //     -
      //       var: "y"
      //       num_bins: 10
      //       clamp: 0
      //     -
      //       var: "z"
      //       num_bins: 10
      //       clamp: 10
      //
      // NEW STYLE BINNING:
      // params:
      //   reduction_op: "sum"
      //   reduction_field: "braid"
      //   output_field: "binning"
      //   output_type: "bins"
      //   axes:
      //     -
      //       field: "x"
      //       num_bins: 10
      //       min_val: -10.0
      //       max_val: 10.0
      //       clamp: 1
      //     -
      //       field: "y"
      //       num_bins: 10
      //       clamp: 0
      //     -
      //       field: "z"
      //       num_bins: 10
      //       clamp: 10
      //
      //  1) data binning: reduction `var` or `reduction_field`
      if(names[i] == "var" || names[i] == "reduction_field" )
      {
        fields.insert(child.as_string());
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
                           "by triggers. Please specify the "
                           "trigger actions directly in the "
                           "trigger parameters.";
      } // actions file
    } // is leaf

    if(!is_leaf)
    {
      // if we are an object, run on each child
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
        else if(names[i] == "axes") // axis list (data binning)
        {
          // normal logic applies
          filter_fields(child, fields, info);
        }
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
bool load_actions_file(const std::string &path,
                       int mpi_comm_id,
                       conduit::Node &actions)
{

    int load_ok = 0;
    int rank = 0;
    actions.reset();

#ifdef ASCENT_MPI_ENABLED

    if(mpi_comm_id == -1)
    {
        // report as failure to load
        return false;
    }

    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_id);
    MPI_Comm_rank(mpi_comm, &rank);

#endif

    if(rank == 0)
    {
        try
        {
            actions.load(path);
            load_ok = 1;
        }
        catch(const conduit::Error &e)
        {
            actions.reset();
        }
    }

#ifdef ASCENT_MPI_ENABLED

    Node n_src, n_reduce;
    n_src = load_ok;
    conduit::relay::mpi::sum_all_reduce(n_src,
                                        n_reduce,
                                        mpi_comm);

    load_ok = n_reduce.value();

    if(load_ok == 0)
    {
        return false;
    }

    relay::mpi::broadcast_using_schema(actions, 0, mpi_comm);

#else

    if(load_ok == 0)
    {
        return false;
    }

#endif

    return true;
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



