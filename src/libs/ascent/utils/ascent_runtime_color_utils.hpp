//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef ASCENT_RUNTIME_COLOR_UTILS_HPP
#define ASCENT_RUNTIME_COLOR_UTILS_HPP

#include <ascent_exports.h>
#include <string>

//-----------------------------------------------------------------------------
// Helpers for parsing colors from runtime params.
//-----------------------------------------------------------------------------
namespace ascent
{
namespace runtime
{
namespace filters
{
namespace detail
{

// Parses "#RGB", "#RGBA", "#RRGGBB", or "#RRGGBBAA" (also accepts no '#', and "0x" prefix).
// Returns true on success, otherwise false and sets err_msg.
bool ASCENT_API parse_hex_color_string(const std::string &input,
                            double &r,
                            double &g,
                            double &b,
                            double &a,
                            bool &has_alpha,
                            std::string &err_msg);

} // namespace detail
} // namespace filters
} // namespace runtime
} // namespace ascent

#endif // ASCENT_RUNTIME_COLOR_UTILS_HPP

