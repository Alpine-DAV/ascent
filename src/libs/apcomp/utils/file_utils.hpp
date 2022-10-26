//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <string>
#include <vector>

#include <apcomp/apcomp_config.h>
#include <apcomp/apcomp_exports.h>
// stolen from llnl/conduit

namespace apcomp
{
std::string  APCOMP_API file_path_separator();

void APCOMP_API split_string(const std::string &str,
                  const std::string &sep,
                  std::string &curr,
                  std::string &next);

void APCOMP_API split_string(const std::string &str,
                  char sep,
                  std::vector<std::string> &sv);

void APCOMP_API rsplit_string(const std::string &str,
                   const std::string &sep,
                   std::string &curr,
                   std::string &next);

void APCOMP_API split_path(const std::string &path,
                std::string &curr,
                std::string &next);

void APCOMP_API rsplit_path(const std::string &path,
                 std::string &curr,
                 std::string &next);

std::string APCOMP_API join_path(const std::string &left,
                      const std::string &right);

void APCOMP_API split_file_path(const std::string &path,
                     std::string &curr,
                     std::string &next);

void APCOMP_API split_file_path(const std::string &path,
                     const std::string &sep,
                     std::string &curr,
                     std::string &next);

void APCOMP_API rsplit_file_path(const std::string &path,
                      std::string &curr,
                      std::string &next);

void APCOMP_API rsplit_file_path(const std::string &path,
                      const std::string &sep,
                      std::string &curr,
                      std::string &next);

std::string
APCOMP_API join_file_path(const std::string &left,
               const std::string &right);

bool APCOMP_API is_file(const std::string &path);

bool APCOMP_API is_directory(const std::string &path);

bool APCOMP_API remove_file(const std::string &path);

bool APCOMP_API remove_directory(const std::string &path);

bool APCOMP_API create_directory(const std::string &path);

} // namespace apcomp
