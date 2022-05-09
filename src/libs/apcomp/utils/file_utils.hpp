#include <string>
#include <vector>

// stolen from llnl/conduit

namespace apcomp
{
std::string  file_path_separator();

void split_string(const std::string &str,
                  const std::string &sep,
                  std::string &curr,
                  std::string &next);

void split_string(const std::string &str,
                  char sep,
                  std::vector<std::string> &sv);

void rsplit_string(const std::string &str,
                   const std::string &sep,
                   std::string &curr,
                   std::string &next);

void split_path(const std::string &path,
                std::string &curr,
                std::string &next);

void rsplit_path(const std::string &path,
                 std::string &curr,
                 std::string &next);

std::string join_path(const std::string &left,
                      const std::string &right);

void split_file_path(const std::string &path,
                     std::string &curr,
                     std::string &next);

void split_file_path(const std::string &path,
                     const std::string &sep,
                     std::string &curr,
                     std::string &next);

void rsplit_file_path(const std::string &path,
                      std::string &curr,
                      std::string &next);

void rsplit_file_path(const std::string &path,
                      const std::string &sep,
                      std::string &curr,
                      std::string &next);

std::string
join_file_path(const std::string &left,
               const std::string &right);

bool is_file(const std::string &path);

bool is_directory(const std::string &path);

bool remove_file(const std::string &path);

bool remove_directory(const std::string &path);

bool create_directory(const std::string &path);

} // namespace apcomp
