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
#include <ascent_runtime_utils.hpp>

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

std::vector<std::string> split(const std::string &s, char delim)
{
  std::vector<std::string> elems;
  detail::split_string(s, delim, elems);
  return elems;
}

//-----------------------------------------------------------------------------

std::regex get_format_pattern(const std::string& variable_name)
{
    // Patterns to identify keyword specified formatting
    //
    // Pattern is '{keyword:format}' where keyword is cycle, family or time and format is a valid
    // integer or floating point standard format.
    //
    // Format loosely defined here any any combination of digits, characters, or a period.
    // More specific formatting constraints are defined in 'expand_generic_variable()'
    std::regex fmt_pattern(R"(\{)" + variable_name + R"((?::([a-zA-Z0-9.]*))?\})");
    return fmt_pattern;
}

template<typename T>
std::string expand_format_value(const std::string path_string,
                                const T value)
{
  std::string result_string = path_string;

  // Maintaining legacy family string formatting
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
                                    const std::string& variable_name, 
                                    const T value,
                                    const std::string& default_formatting)
{
    std::smatch match;
    std::string result_string = path_string;

    // Defining a valid int format to be any number of digits followed by a d,i, or u character
    //
    // The different supported integer number types are:
    //    d: signed decimal integer
    //    i: signed decimal integer
    //    u: unsigned decimal integer
    //
    // e.g. 003d or 34i or 1d or 4u
    std::regex valid_int_format(R"(^(\d*)?[diu]$)");

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
    std::regex valid_float_format(R"(^(\d*(\.\d+)?)?[fFeEgG]$)");

    while (std::regex_search(result_string, match, get_format_pattern(variable_name)))
    {
        std::string format_spec = match[1].str();

        // Handle cases where the default formatting should be used
        // These are either when no format or an invalid format have been given
        if (format_spec.size() == 0)
        {
            format_spec = default_formatting;
        }
        else if (!std::regex_match(format_spec, valid_int_format) && !std::regex_match(format_spec, valid_float_format))
        {
            ASCENT_WARN("Invalid format specifier: '" 
                        << format_spec 
                        << "'. Inserting value with default formatting of %" 
                        << default_formatting);
            format_spec = default_formatting;
        }

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
    }

    return result_string;
}

int check_directory_for_family_value(const std::string& path_string,
                                     int mpi_comm_id,
                                     int family_value)
{
  // Initialized the MPI variables if needed
  int rank = 0;
#ifdef ASCENT_MPI_ENABLED
  if(mpi_comm_id == -1)
  {
      // do nothing, an error will be thrown later
      // so we can respect the exception handling
      return family_value;
  }

  MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_id);
  MPI_Comm_rank(mpi_comm, &rank);
#endif

  // Determining the file name and directory name
  std::string file_name_fmt, dir_path;
  conduit::utils::rsplit_file_path(path_string, file_name_fmt, dir_path);
  if (dir_path.size() == 0) {
    dir_path = ".";
  }


  // This pattern is used to match to numbers.
  // It is looking for integers, decimal, and scientific notation values
  // Explanation:
  //      [+-]?              - Optional + or - symbol
  //      \d+                - At least one digit
  //      (:?\.\d+)?         - Optional decimal value in a non-capturing group
  //      (?:[eE][+-]?\d+)?  - Optional handling for scientific notation (e.g. e+02)
  std::string number_pattern = R"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)";
  
  std::string search_pattern_str = file_name_fmt;
  std::smatch match;
  if (std::regex_search(search_pattern_str, match, get_format_pattern("family")))
  {
    // This adds a capturing group that matches to a decimal number
    // When running a regex_search this value will be captured and saved out
    search_pattern_str.replace(match.position(0), match.length(0), "(" + number_pattern + ")");
  }

  while (std::regex_search(search_pattern_str, match, get_format_pattern(R"([a-zA-Z]*)")))
  {
    // This adds a the pattern for a decimal number
    search_pattern_str.replace(match.position(0), match.length(0), number_pattern);
  }

  // Use the defined pattern to make the final regular expression
  // Adding a ^ to lock the pattern to the start of the file_name and a pattern for a file extension
  // at the end to the pattern
  search_pattern_str = "^" + search_pattern_str + R"($)";
  std::regex search_pattern(search_pattern_str);

  if (rank == 0) {
    // Checking the directory contents for any filenames that match
    std::vector<std::string> contents;
    conduit::utils::list_directory_contents(dir_path, contents);

    for (const std::string& item : contents)
    {
      std::string file_name, rm_path;
      conduit::utils::rsplit_file_path(item, file_name, rm_path);
      
      std::smatch file_match;
      if (std::regex_search(file_name, file_match, search_pattern))
      {
        std::string match_string = file_match[1].str();
        
        // If a family value stored in the file name, check it and update the family value if needed
        if (match_string.size() != 0)
        {
            int matched_value = static_cast<int>(std::stod(match_string));
            // If we find a match that is greater than the current family value, update to a new value
            if (matched_value >= family_value)
            {
                family_value = matched_value + 1;
            }
        }
      }
    }
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Bcast(&family_value, 1, MPI_INT, 0, mpi_comm);
#endif

  return family_value;
}

int get_family_value(const std::string& path_string, 
                     const std::string& file_extension,
                     int mpi_comm_id,
                     int family_value)
{
  std::string modified_path_string = path_string;

  // If extension not added to the file path, add it
  std::string file_name, ext;
  conduit::utils::rsplit_string(modified_path_string, ".", ext, file_name);
  if (file_extension.compare("."+ext) != 0) {
    modified_path_string += file_extension;
  }

  // Check the file directory to determine a valid family value. Increases the value if needed.
  family_value = check_directory_for_family_value(modified_path_string,
                                                  mpi_comm_id,
                                                  family_value);

  static std::map<std::string, int> s_file_family_map;
  bool exists = s_file_family_map.find(modified_path_string) != s_file_family_map.end();

  if(!exists)
  {
    s_file_family_map[modified_path_string] = family_value;
  }
  else
  {
    family_value = s_file_family_map[modified_path_string] + 1;
    s_file_family_map[modified_path_string] = family_value;
  }

  return family_value;
}

std::string expand_path_special_variables(const std::string &path_string,
                                          const std::string &file_extension,
                                          int mpi_comm_id,
                                          bool append_if_no_format)
{
    std::regex invalid_pattern(R"(\{([a-zA-Z]*):.*\})");

    std::smatch match;
    if (std::regex_search(path_string, match, invalid_pattern))
    {
      std::string keyword = match[1].str();
      if (keyword != "cycle" && keyword != "family" && keyword != "time")
      {
        ASCENT_WARN("Invalid format keyword '"
                    << match[1].str()
                    << "'. Only cycle, family, and time are supported");
      }
    }

    std::string result_string = ascent::runtime::filters::output_dir(path_string);

    conduit::Node meta = Metadata::n_metadata;

    // If no formatting has been added to the path, add one to the end of the path
    // This will either add the cycle value if cycle is available or the family value if not
    if (append_if_no_format
        && !std::regex_search(result_string, get_format_pattern(R"([a-zA-Z]*)")) 
        && !std::regex_search(result_string, std::regex(R"(%(\d*)?(\.\d+)?[a-zA-Z])")))
    {
        std::stringstream ss;
        ss << result_string 
           << (result_string.back() != '_' ? "_" : "") 
           << (meta.has_path("cycle") ? "{cycle}" : "{family}");
        result_string = ss.str();
    }

    int family_value = 0;
    if (meta.has_path("family_value_seed"))
    {
        family_value = meta["family_value_seed"].to_value();
    }
    family_value = get_family_value(path_string, file_extension, mpi_comm_id, family_value);
    result_string = expand_generic_variable(result_string, "family", family_value, "06d");

    int cycle = 0;
    if (meta.has_path("cycle"))
    {
        cycle = meta["cycle"].to_value();
        result_string = expand_generic_variable(result_string, "cycle", cycle, "06d");
        result_string = expand_format_value(result_string, cycle);
    }
    
    if (meta.has_path("time"))
    {
        float time = meta["time"].to_value();
        result_string = expand_generic_variable(result_string, "time", time, "g");
    }

    return result_string;
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



