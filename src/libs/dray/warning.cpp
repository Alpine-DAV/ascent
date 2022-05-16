// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/warning.hpp>
#include <iostream>

namespace dray
{

void warning(const std::string message, const std::string file, int line)
{
  if(dray::mpi_rank() == 0)
  {
    std::stringstream msg;
    msg<<file<<" ("<<line<<"): "<<message<<"\n";
    std::cerr<<msg.str();
  }
}

} // namespace dray
