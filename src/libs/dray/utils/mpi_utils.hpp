// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef DRAY_MPI_UTILS_HPP
#define DRAY_MPI_UTILS_HPP

#include <set>
#include <string>
namespace dray
{

//
// returns true if all ranks say true
//
bool global_agreement(bool vote);

//
// returns true if any ranks says true
//
bool global_someone_agrees(bool vote);

//
// gathers strings from all ranks
//
void gather_strings(std::set<std::string> &set);

};

#endif


