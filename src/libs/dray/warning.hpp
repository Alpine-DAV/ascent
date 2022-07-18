// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_WARNING_HPP
#define DRAY_WARNING_HPP

#include <string>
#include <sstream>

namespace dray
{

void warning(const std::string message, const std::string file, int line);

#define DRAY_WARNING( msg )                     \
{                                               \
    std::ostringstream oss_error;               \
    oss_error << msg;                           \
    warning(oss_error.str(),                    \
            std::string(__FILE__),              \
            __LINE__);                          \
}                                               \

} // namespace dray
#endif
