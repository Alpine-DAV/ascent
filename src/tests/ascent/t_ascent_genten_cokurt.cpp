//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_ascent_genten_cokurt.cpp
///
//-----------------------------------------------------------------------------


#include "gtest/gtest.h"

#include <ascent.hpp>

#include <iostream>
#include <math.h>

#include <conduit_blueprint.hpp>

#include "t_config.hpp"
#include "t_utils.hpp"

#include <vector>

using namespace std;
using namespace conduit;
using namespace ascent;


using namespace std;
using namespace conduit;
using namespace ascent;


index_t EXAMPLE_MESH_SIDE_DIM = 20;

//-----------------------------------------------------------------------------
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
    string output_file = conduit::utils::join_file_path(output_path,"tout_genten_spatial_metric");

   // remove old images before rendering
    remove_test_image(output_file);

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"] = "learn";
    extracts["e1/params/threshold"] = 0.1;
    extracts["e1/params/fields"].append() = "var1";
    extracts["e1/params/fields"].append() = "var2";
    extracts["e1/params/path"] = output_file;

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


//-----------------------------------------------------------------------------
TEST(ascent_cokurt, test_single_domain)
{
    Node mesh;
    conduit::blueprint::mesh::examples::braid("uniform",
                                              20,
                                              20,
                                              20,
                                              mesh);

    mesh["fields/braid2"].set(mesh["fields/braid"]);

    string output_path = prepare_output_dir();
    string output_file = conduit::utils::join_file_path(output_path,
                                                        "tout_genten_single_domain_spatial_metric");

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];
    extracts["e1/type"] = "learn";
    extracts["e1/params/threshold"] = 0.1;
    extracts["e1/params/fields"].append() = "braid";
    extracts["e1/params/fields"].append() = "braid2";
    extracts["e1/params/path"] = output_file;
    //
    // Run Ascent
    //

    Ascent ascent;
    ascent.open();
    ascent.publish(mesh);
    ascent.execute(actions);
    ascent.close();
}
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    if(argc == 2)
    {
        EXAMPLE_MESH_SIDE_DIM = atoi(argv[1]);
    }

    result = RUN_ALL_TESTS();
    return result;
}


