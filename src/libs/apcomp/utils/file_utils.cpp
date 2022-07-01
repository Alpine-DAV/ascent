//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <apcomp/utils/file_utils.hpp>
// file system funcs
#include <sys/stat.h>
#include <sys/types.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <limits>
#include <fstream>

namespace apcomp
{

std::string
file_path_separator()
{
  // TODO: add windows support
  return "/";
}

void
split_string(const std::string &str,
             const std::string &sep,
             std::string &curr,
             std::string &next)
{
  curr.clear();
  next.clear();

  std::size_t found = str.find(sep);
  if (found != std::string::npos)
  {
    curr = str.substr(0,found);
    if(found != str.size()-1)
        next = str.substr(found+1,str.size()-(found-1));
  }
  else
  {
    curr = str;
  }
}

void
split_string(const std::string &str, char sep, std::vector<std::string> &sv)
{
  if(!str.empty())
  {
    const char *start = str.c_str();
    const char *c     = str.c_str();
    while(*c != '\0')
    {
      if(*c == sep)
      {
        size_t len = c - start;
        if(len > 0)
            sv.push_back(std::string(start, len));
        c++;
        start = c;
      }
      else
        c++;
    }
    if(*start != '\0')
    {
      size_t len = c - start;
      if(len > 0)
        sv.push_back(std::string(start, len));
    }
  }
}

void
rsplit_string(const std::string &str,
              const std::string &sep,
              std::string &curr,
              std::string &next)
{
  curr.clear();
  next.clear();

  std::size_t found = str.rfind(sep);
  if (found != std::string::npos)
  {
    next = str.substr(0,found);
    if(found != str.size()-1)
      curr = str.substr(found+1,str.size()-(found-1));
  }
  else
  {
    curr = str;
  }
}

void
split_path(const std::string &path,
           std::string &curr,
           std::string &next)
{
  split_string(path,
               std::string("/"),
               curr,
               next);
}

void
rsplit_path(const std::string &path,
            std::string &curr,
            std::string &next)
{
  rsplit_string(path,
                std::string("/"),
                curr,
                next);
}

std::string
join_path(const std::string &left,
          const std::string &right)
{
  std::string res = left;
  if(res.size() > 0 &&
     res[res.size()-1] != '/' &&
     right.size() > 0 )
  {
      res += "/";
  }
  res += right;
  return res;
}

void
split_file_path(const std::string &path,
                std::string &curr,
                std::string &next)
{
  split_string(path,
               file_path_separator(),
               curr,
               next);
}

void
split_file_path(const std::string &path,
                const std::string &sep,
                std::string &curr,
                std::string &next)
{
  // if we are splitting by ":", we need to be careful on windows
  // since drive letters include ":"
  //
  // NOTE: We could if-def for windows, but its nice to be able
  // to run unit tests on other platforms.
  if( sep == std::string(":") &&
      path.size() > 2 &&
      path[1] == ':' &&
      path[2] == '\\')
  {
    // eval w/o drive letter
    if(path.size() > 3)
    {
      std::string check_path = path.substr(3);
      split_string(check_path,
                   sep,
                   curr,
                   next);
      // add drive letter back
      curr = path.substr(0,3) + curr;
    }
    else
    {
      // degen case, we we only have the drive letter
      curr = path;
      next = "";
    }
  }
  else
  {
    // normal case
    split_string(path,
                 sep,
                 curr,
                 next);

  }
}

void
rsplit_file_path(const std::string &path,
                 std::string &curr,
                 std::string &next)
{
  rsplit_string(path,
                file_path_separator(),
                curr,
                next);
}
void
rsplit_file_path(const std::string &path,
                 const std::string &sep,
                 std::string &curr,
                 std::string &next)
{
  // if we are splitting by ":", we need to be careful on windows
  // since drive letters include ":"
  //
  // NOTE: We could if-def for windows, but its nice to be able
  // to run unit tests on other platforms.
  if( sep == std::string(":") &&
      path.size() > 2 &&
      path[1] == ':' &&
      path[2] == '\\')
  {
    // eval w/o drive letter
    if(path.size() > 3)
    {
      std::string check_path = path.substr(3);
      rsplit_string(check_path,
                    sep,
                    curr,
                    next);
      // add drive letter back
      if(next == "")
      {
        // there was no split
        curr = path.substr(0,3) + curr;
      }
      else
      {
        // there was a split
        next = path.substr(0,3) + next;
      }
    }
    else
    {
      // degen case, we we only have the drive letter
      curr = path;
      next = "";
    }
  }
  else
  {
     // normal case
    rsplit_string(path,
                  sep,
                  curr,
                  next);

  }
}

std::string
join_file_path(const std::string &left,
               const std::string &right)
{
  std::string res = left;
  if(res.size() > 0 && res[res.size()-1] != file_path_separator()[0])
  {
      res += file_path_separator();;
  }
  res += right;
  return res;
}

bool
is_file(const std::string &path)
{
  bool res = false;
  struct stat path_stat;
  if(stat(path.c_str(), &path_stat) == 0)
  {
      if(path_stat.st_mode & S_IFREG)
          res = true;
  }
  return res;
}

bool
is_directory(const std::string &path)
{
  bool res = false;
  struct stat path_stat;
  if (stat(path.c_str(), &path_stat) == 0)
  {
    if (path_stat.st_mode & S_IFDIR)
        res = true;
  }
  return res;
}

bool
remove_file(const std::string &path)
{
  return ( remove(path.c_str()) == 0 );
}

bool
remove_directory(const std::string &path)
{
  return ( remove(path.c_str()) == 0 );
}

bool
create_directory(const std::string &path)
{
  return (mkdir(path.c_str(),S_IRWXU | S_IRWXG) == 0);
}

} // namespace apcomp
