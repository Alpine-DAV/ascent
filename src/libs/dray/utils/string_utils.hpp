// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef DRAY_STRING_UTILS_HPP
#define DRAY_STRING_UTILS_HPP

#include <string>
#include <vector>

namespace dray
{

std::string expand_family_name(const std::string name, int counter = 0);

std::vector<std::string> split(const std::string &s, char delim = ' ');

bool contains(std::vector<std::string> &names, const std::string name);

std::string timestamp();

}; // namespace dray

#endif


