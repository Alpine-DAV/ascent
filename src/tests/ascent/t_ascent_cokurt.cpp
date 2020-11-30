//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <vector>
#include <math.h>

#include "t_config.hpp"


using namespace std;
using namespace conduit;
using namespace ascent;

void create_uniform_domain(conduit::Node &dom,
                           double x_min,
                           double x_max,
                           double y_min,
                           double y_max,
                           double z_min,
                           double z_max,
                           int res)
{
  dom["topologies/topo/coordset"] = "coords";
  dom["topologies/topo/type"] = "uniform";

  dom["coordsets/coords/type"] = "uniform";
  dom["coordsets/coords/dims/i"] = res + 1;
  dom["coordsets/coords/dims/j"] = res + 1;
  dom["coordsets/coords/dims/k"] = res + 1;
  dom["coordsets/coords/origin/x"] = x_min;
  dom["coordsets/coords/origin/y"] = y_min;
  dom["coordsets/coords/origin/z"] = z_min;
  double dx = (x_max - x_min) / (double)res;
  double dy = (y_max - y_min) / (double)res;
  double dz = (z_max - z_min) / (double)res;
  dom["coordsets/coords/spacing/dx"] = dx;
  dom["coordsets/coords/spacing/dy"] = dy;
  dom["coordsets/coords/spacing/dz"] = dz;

  int n_pts = res + 1;
  // we store x varying the fastest
  for(int z = 0; z < n_pts; ++z)
  {
    double p_z = z_min + dz * z;
    for(int y = 0; y < n_pts; ++y)
    {
      double p_y = y_min + dy * y;
      for(int x = 0; x < n_pts; ++x)
      {
        // point = (p_x, p_y, p_z)
        double p_x = x_min + dx * x;
      }
    }
  }

}

void build_data_set(conduit::Node &dataset)
{
  int domain_res = 10; // each domain will be domain_res^3 zones
  double xyz_min = -10;
  double xyz_max = 10;
  // we will create a data set with base^3 domains
  const int32 base = 2;
  std::vector<double> divs;
  divs.resize(base+1);
  double delta = (xyz_max - xyz_min) / (double) base;
  for(int i = 0; i < base; ++i)
  {
    divs[i] = i * delta + xyz_min;
  }
  // account for any floating point roundoff
  divs[base] = xyz_max;

  for(int x = 0; x < base; x++)
  {
    for(int y = 0; y < base; y++)
    {
      for(int z = 0; z < base; z++)
      {
        create_uniform_domain(dataset.append(),
                              divs[x], divs[x+1],
                              divs[y], divs[y+1],
                              divs[z], divs[z+1],
                              domain_res);
      }
    }
  }
}

//-----------------------------------------------------------------------------
TEST(ascent_smoke, ascent_about)
{
}

