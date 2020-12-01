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

#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_utils.hpp"

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
  std::vector<double> var_1;
  std::vector<double> var_2;
  var_1.resize(n_pts * n_pts * n_pts);
  var_2.resize(n_pts * n_pts * n_pts);

  // we store x varying the fastest
  int counter = 0;
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
        // sphere 1 = (-4,0,0) r = 6
        // sphere 2 = (4,0,0)  r = 6
        double r1 = sqrt((p_x - 4) * (p_x - 4) + p_y * p_y + p_z * p_z) > 6 ? 0.5 : 1.f;
        double r2 = sqrt((p_x + 4) * (p_x + 4) + p_y * p_y + p_z * p_z) > 6 ? 0.5 : 1.0;
        //std::cout<<" r1 "<<r1<<" "<<p_x<<" "<<p_y<<" "<<p_z<<"\n";
        var_1[counter] = r1;
        var_2[counter] = r2;
        counter++;
      }
    }
  }
 dom["fields/var1/association"] = "vertex";
 dom["fields/var1/type"]        = "scalar";
 dom["fields/var1/topology"]    = "topo";
 dom["fields/var1/values"].set(var_1);

 dom["fields/var2/association"] = "vertex";
 dom["fields/var2/type"]        = "scalar";
 dom["fields/var2/topology"]    = "topo";
 dom["fields/var2/values"].set(var_2);

}

void build_data_set(conduit::Node &dataset)
{
  int domain_res = 64; // each domain will be domain_res^3 zones
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
TEST(ascent_cokurt, test_field)
{
    conduit::Node data;
    build_data_set(data);
    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,"tout_marco");

    // remove old images before rendering
    remove_test_image(output_file);

    conduit::Node actions;
    conduit::Node pipelines;
    // pipeline 1
    pipelines["pl1/f1/type"] = "slice";
    // filter knobs
    conduit::Node &slice_params = pipelines["pl1/f1/params"];
    slice_params["point/x"] = 0.f;
    slice_params["point/y"] = 0.f;
    slice_params["point/z"] = 0.f;

    slice_params["normal/x"] = 0.f;
    slice_params["normal/y"] = 0.f;
    slice_params["normal/z"] = 1.f;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"]         = "pseudocolor";
    scenes["s1/plots/p1/field"] = "var2";
    scenes["s1/plots/p1/pipeline"] = "pl1";
    scenes["s1/image_prefix"] = output_file;

    // add the pipeline
    conduit::Node &add_pipelines = actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;
    // add the scenes
    conduit::Node &add_scenes= actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;

    //
    // Run Ascent
    //

    Ascent ascent;

    Node ascent_opts;
    ascent_opts["runtime/type"] = "ascent";
    ascent.open(ascent_opts);
    ascent.publish(data);
    ascent.execute(actions);
    ascent.close();
}

