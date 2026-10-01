//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ascent_runtime_color_utils.hpp"

#include <cctype>

namespace
{

int hex_digit_value(char c)
{
  if(c >= '0' && c <= '9')
  {
    return c - '0';
  }
  if(c >= 'a' && c <= 'f')
  {
    return 10 + (c - 'a');
  }
  if(c >= 'A' && c <= 'F')
  {
    return 10 + (c - 'A');
  }
  return -1;
}

std::string trim_copy(const std::string &s)
{
  std::size_t begin = 0;
  while(begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])))
  {
    ++begin;
  }

  std::size_t end = s.size();
  while(end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])))
  {
    --end;
  }

  return s.substr(begin, end - begin);
}

} // namespace

namespace ascent
{
namespace runtime
{
namespace filters
{
namespace detail
{

bool parse_hex_color_string(const std::string &input,
                            double &r,
                            double &g,
                            double &b,
                            double &a,
                            bool &has_alpha,
                            std::string &err_msg)
{
  std::string s = trim_copy(input);
  if(s.empty())
  {
    err_msg = "empty string";
    return false;
  }

  if(s.size() >= 2 && (s[0] == '0') && (s[1] == 'x' || s[1] == 'X'))
  {
    s = s.substr(2);
  }

  if(!s.empty() && s[0] == '#')
  {
    s = s.substr(1);
  }

  const std::size_t n = s.size();
  if(!(n == 3 || n == 4 || n == 6 || n == 8))
  {
    err_msg = "expected 3, 4, 6, or 8 hex digits";
    return false;
  }

  auto read_nibble = [&](std::size_t idx, int &out) -> bool {
    const int v = hex_digit_value(s[idx]);
    if(v < 0)
    {
      err_msg = "invalid hex digit";
      return false;
    }
    out = v;
    return true;
  };

  auto read_byte_from_pair = [&](std::size_t idx, int &out) -> bool {
    int hi = 0;
    int lo = 0;
    if(!read_nibble(idx, hi) || !read_nibble(idx + 1, lo))
    {
      return false;
    }
    out = (hi << 4) | lo;
    return true;
  };

  int rb = 0;
  int gb = 0;
  int bb = 0;
  int ab = 255;
  has_alpha = (n == 4 || n == 8);

  if(n == 3 || n == 4)
  {
    int rn = 0;
    int gn = 0;
    int bn = 0;
    int an = 15;
    if(!read_nibble(0, rn) || !read_nibble(1, gn) || !read_nibble(2, bn))
    {
      return false;
    }
    if(n == 4 && !read_nibble(3, an))
    {
      return false;
    }
    rb = rn * 17;
    gb = gn * 17;
    bb = bn * 17;
    ab = an * 17;
  }
  else
  {
    if(!read_byte_from_pair(0, rb) || !read_byte_from_pair(2, gb) || !read_byte_from_pair(4, bb))
    {
      return false;
    }
    if(n == 8 && !read_byte_from_pair(6, ab))
    {
      return false;
    }
  }

  r = static_cast<double>(rb) / 255.0;
  g = static_cast<double>(gb) / 255.0;
  b = static_cast<double>(bb) / 255.0;
  a = static_cast<double>(ab) / 255.0;
  return true;
}

} // namespace detail
} // namespace filters
} // namespace runtime
} // namespace ascent

