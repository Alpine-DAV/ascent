
#include <dray/utils/string_utils.hpp>
#include <map>
#include <ctime>
#include <sstream>
#include <stdio.h>
#include <algorithm>


namespace dray
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

bool contains(std::vector<std::string> &names, const std::string name)
{
  if (std::find(names.begin(), names.end(), name) != names.end())
  {
    return true;
  }
  return false;
}

std::string expand_family_name(const std::string name, int counter)
{
  if(counter == 0)
  {
    static std::map<std::string, int> s_file_family_map;
    bool exists = s_file_family_map.find(name) != s_file_family_map.end();
    if(!exists)
    {
      s_file_family_map[name] = counter;
    }
    else
    {
      counter = s_file_family_map[name];
      s_file_family_map[name] = counter + 1;
    }
  }

  std::string result;
  bool has_format = name.find("%") != std::string::npos;
  if(has_format)
  {
    // allow for long file paths
    char buffer[1000];
    sprintf(buffer, name.c_str(), counter);
    result = std::string(buffer);
  }
  else
  {
    std::stringstream ss;
    ss<<name<<counter;
    result = ss.str();
  }
  return result;
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
}; // namespace dray



