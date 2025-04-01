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
#include <ascent.hpp>
#include <ascent_metadata.hpp>

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
std::string expand_format_value(const std::string path_string,
                                const T value)
{
  std::string result_string = path_string;

  // Maintaing legacy family string formatting
  bool has_format = result_string.find("%") != std::string::npos;
  if(has_format)
  {
    // allow for long file paths
    char buffer[2048];
    snprintf(buffer, 2048, result_string.c_str(), value);
    result_string = std::string(buffer);
  }

  return result_string;
}

template<typename T>
std::string expand_generic_variable(const std::string& path_string, 
                                    const std::regex& pattern, 
                                    const T value)
{
    std::smatch match;
    std::string result_string = path_string;

    // Defining a valid int format to be any number of digits followed by a d,i, or u character
    //
    // The different supported integer number types are:
    //    d: signed decimal integer
    //    i: digned decimal integer
    //    u: unsigned decimal integer
    //
    // e.g. 003d or 34i or 1d or 4u
    std::regex valid_int_format(R"(^\d*[diu]$)");

    // Defining a valid float format to be a real number followed by a f,e,g,F,E, or G character
    // Real numbers are any number of digits potentially followed by a decimal point and at least one digit
    //
    // The different supported floating point number types are:
    //    f: decimal floating point
    //    F: decimal floating point
    //    e: scientific notation using e to indicate the exponent
    //    E: same as e except uses E to indicate the exponent
    //    g: Uses the shortest notation (either e or f)
    //    G: Uses the shortest notation (either E or F)
    //
    // e.g. 3.14f or 2F or 023g or 10.4e
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
        else if (format_spec.size() == 0) {
          ASCENT_WARN("No format specifications given. Inserting value without formatting.");
          result_string.replace(match.position(0), match.length(0), std::to_string(value));
        }
        else
        {
          ASCENT_WARN("Invalid format specifier: '" 
                        << format_spec 
                        << "'. Inserting value without formatting.");
          result_string.replace(match.position(0), match.length(0), std::to_string(value));
        }
    }

    return result_string;
}

int check_directory_for_family_value(const std::string& path_string, int family_value) {

  // Determining the file name and directory name
  std::string file_name_fmt, dir_path;
  conduit::utils::rsplit_file_path(path_string, file_name_fmt, dir_path);
  if (dir_path.size() == 0) {
    dir_path = ".";
  }

  // Builting a pattern to match to filenames
  std::regex family_pattern(R"(\{family:([a-zA-Z0-9.]*)\})");
  std::regex other_fmts_pattern(R"(\{[a-zA-Z]*:[a-zA-Z0-9.]*\})");
  std::string search_pattern_str = "^" + file_name_fmt + ".*";
  std::smatch match;
  if (std::regex_search(search_pattern_str, match, family_pattern))
  {
    search_pattern_str.replace(match.position(0), match.length(0), "([0-9.]*)");
  }
  while (std::regex_search(search_pattern_str, match, other_fmts_pattern))
  {
    search_pattern_str.replace(match.position(0), match.length(0), "[0-9.]*");
  }
  std::regex search_pattern(search_pattern_str);

  // Checking the directory contents for any filenames that match
  std::vector<std::string> contents;
  conduit::utils::list_directory_contents(dir_path, contents);
  for (const std::string& item : contents) {
      std::string file_name, rm_path;
      conduit::utils::rsplit_file_path(item, file_name, rm_path);
      
      std::smatch file_match;
      if (std::regex_search(file_name, file_match, search_pattern)) {
        int matched_value = std::stoi(file_match[1].str());
        if (matched_value > family_value)
        {
          family_value = matched_value + 1;
        }
      }
  }

  return family_value;
}

int get_family_value(const std::string& path_string, int family_value)
{
  
  std::string modified_path_string = path_string;

  int rank = 0;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &rank);
#endif

  if (rank == 0)
  {
    family_value = check_directory_for_family_value(path_string, family_value);
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Bcast(&family_value, 1, MPI_INT, 0, mpi_comm);
#endif

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

  return family_value;
}

std::string expand_path_special_variables(const std::string &path_string,
                                          int counter,
                                          bool append_if_no_format)
{
    // Parterns to identify keyword specified formatting
    //
    // Pattern is '{keyword:format}' where keyword is cycle, family or time and format is a valid
    // integer or floating point standard format.
    //
    // Format loosly defined here any any combination of digits, characters, or a period.
    // More specific formatting constraints are defined in 'expand_generic_variable()'
    std::regex cycle_pattern(R"(\{cycle:([a-zA-Z0-9.]*)\})");
    std::regex family_pattern(R"(\{family:([a-zA-Z0-9.]*)\})");
    std::regex time_pattern(R"(\{time:([a-zA-Z0-9.]*)\})");
    std::regex invalid_pattern(R"(\{([a-zA-Z]*):.*\})");

    std::smatch match;
    if (std::regex_search(path_string, match, invalid_pattern)) {
      std::string keyword = match[1].str();
      if (keyword != "cycle" && keyword != "family" && keyword != "time") {
        ASCENT_WARN("Invalid format keyword '"
                    << match[1].str()
                    << "'. Only cycle, family, and time are supported");
      }
    }

    std::string result_string = path_string;

    conduit::Node meta = Metadata::n_metadata;

    int family_value = get_family_value(path_string, counter);
    result_string = expand_generic_variable(result_string, family_pattern, family_value);
    result_string = expand_format_value(result_string, family_value);

    if (meta.has_path("cycle")) {
        int cycle = meta["cycle"].to_value();
        result_string = expand_generic_variable(result_string, cycle_pattern, cycle);
    }
    
    if (meta.has_path("time")) {
        float time = meta["time"].to_value();
        result_string = expand_generic_variable(result_string, time_pattern, time);
    }

    if (result_string == path_string && append_if_no_format) {
        std::stringstream ss;
        ss<<result_string<<family_value;
        result_string = ss.str();
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



