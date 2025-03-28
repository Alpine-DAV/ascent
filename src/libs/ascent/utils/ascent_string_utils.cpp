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

#include "ascent_string_utils.hpp"
#include <map>
#include <ctime>
#include <sstream>
#include <stdio.h>
#include <regex>
#include <conduit.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#include <flow.hpp>
#endif


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

namespace detail
{
void split_string(const std::string &s,
                  char delim,
                  std::vector<std::string> &elems)
{
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim))
  {
    elems.push_back(item);
  }
}

} // namespace detail

template<typename T>
std::string expand_generic_variable(const std::string& path_string, 
                                    const std::regex& pattern, 
                                    const T value)
{
    std::smatch match;
    std::string result_string = path_string;

    std::regex valid_int_format(R"(^\d*d$)");
    std::regex valid_float_format(R"(^\d*(\.\d+)?[fFeEgG]$)");

    while (std::regex_search(result_string, match, pattern))
    {
        std::string format_spec = match[1].str();

        if (std::regex_match(format_spec, valid_int_format))
        {
          char formatted_number[50];
          std::string full_format = "%" + format_spec;
          snprintf(formatted_number, sizeof(formatted_number), full_format.c_str(), static_cast<int>(value));
          result_string.replace(match.position(0), match.length(0), formatted_number);
        }
        else if (std::regex_match(format_spec, valid_float_format))
        {
          char formatted_number[50];
          std::string full_format = "%" + format_spec;
          snprintf(formatted_number, sizeof(formatted_number), full_format.c_str(), static_cast<float>(value));
          result_string.replace(match.position(0), match.length(0), formatted_number);
        }
        else
        {
          // std::cout<< "Invalid format specifier: " << format_spec << std::end;
          // Fallback: use default string conversion.
          result_string.replace(match.position(0), match.length(0), std::to_string(value));
          
        }
    }

    return result_string;
}

std::string expand_family_variable(const std::string& path_string, 
                                   int family_value)
{
  std::regex family_pattern(R"(\{family:((\d+\.)?\d+\D)\})");
  std::string modified_path_string = path_string;

  int rank = 0;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);
#endif

  if(rank == 0)
  {
    static std::map<std::string, int> s_file_family_map;
    bool exists = s_file_family_map.find(path_string) != s_file_family_map.end();
    if(!exists)
    {
      s_file_family_map[path_string] = family_value;
    }
    else
    {
      family_value = s_file_family_map[path_string] + 1;
      s_file_family_map[path_string] = family_value;
    }
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Bcast(&family_value, 1, MPI_INT, 0, mpi_comm);
#endif

  // Maintaing legacy family string formatting
  bool has_format = modified_path_string.find("%") != std::string::npos;
  if(has_format)
  {
    // allow for long file paths
    char buffer[2048];
    snprintf(buffer, 2048, modified_path_string.c_str(), family_value);
    modified_path_string = std::string(buffer);
  }

  return expand_generic_variable(modified_path_string, family_pattern, family_value);
}

std::string expand_path_special_variables(const std::string path_string, 
                                          const conduit::Node &meta,
                                          int counter)
{
    std::regex cycle_pattern(R"(\{cycle:((\d+\.)?\d+\D)\})");
    std::regex time_pattern(R"(\{time:((\d+\.)?\d+\D)\})");

    std::cout << "This is the metadata right now" << std::endl;
    meta.print();
    
    std::string result_string = path_string;

    result_string = expand_family_variable(result_string, counter);

    if (meta.has_path("cycle")) {
        int cycle = meta["cycle"].to_value();
        result_string = expand_generic_variable(result_string, cycle_pattern, cycle);
    }
    
    if (meta.has_path("time")) {
        float time = meta["time"].to_value();
        result_string = expand_generic_variable(result_string, time_pattern, time);
    }

    return result_string;
}

std::vector<std::string> split(const std::string &s, char delim)
{
  std::vector<std::string> elems;
  detail::split_string(s, delim, elems);
  return elems;
}

//-----------------------------------------------------------------------------
std::string
timestamp()
{
    // create std::string that reps current time
    time_t t;
    tm *t_local;
    time(&t);
    t_local = localtime(&t);
    char buff[256];
    strftime(buff, sizeof(buff), "%Y-%m-%d %H:%M:%S", t_local);
    return std::string(buff);
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



