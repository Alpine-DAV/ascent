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
/// file: t_ascent_babelfow_pmt_mpi.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <cfloat>
#include <conduit_blueprint.hpp>
#include <mpi.h>
#include <ctime>
#include <cassert>


#include "gtest/gtest.h"

#include <ascent.hpp>
#include <mpi.h>

#include "t_config.hpp"
#include "t_utils.hpp"

using namespace std;
using namespace conduit;
using ascent::Ascent;
using namespace ascent;

typedef double FunctionType;

TEST(ascent_babelfow_rel_mpi, test_babelfow_rel_mpi)
{
  using namespace std;
  int provided;

  int32_t dim = 50;
  clock_t start, finish;

  double run_time, max_run_time;

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  Ascent a;
  conduit::Node ascent_opt;
  ascent_opt["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent_opt["runtime/type"] = "ascent";
  a.open(ascent_opt);

  // user defines n_blocks per dimension
  // user provides the size of the whole data
  vector <int32_t> data_size({dim, dim, dim});
  vector <int32_t> n_blocks({2, 2, 2});
  int32_t block_size[3] = {data_size[0] / n_blocks[0], data_size[1] / n_blocks[1], data_size[2] / n_blocks[2]};
  // compute the boundaries of the needed block
  vector <int32_t> low(3);
  vector <int32_t> high(3);
  low[0] = mpi_rank % n_blocks[0] * block_size[0];
  low[1] = mpi_rank / n_blocks[0] % n_blocks[1] * block_size[1];
  low[2] = mpi_rank / n_blocks[1] / n_blocks[2] % n_blocks[2] * block_size[2];
  high[0] = std::min(low[0] + block_size[0], data_size[0] - 1);
  high[1] = std::min(low[1] + block_size[1], data_size[1] - 1);
  high[2] = std::min(low[2] + block_size[2], data_size[2] - 1);

  // for testing purpose: every rank has whole data
  // in practice, Only assign the corresponding block(s) to each rank
  // The user should define block_data or that should come from the simulation\
  // NOTE: PMT assumes Ghost Layers only in positive x,y,z directions

  // size of the local data
  int32_t num_x = high[0] - low[0] + 1;
  int32_t num_y = high[1] - low[1] + 1;
  int32_t num_z = high[2] - low[2] + 1;
  vector<FunctionType> block_data(num_x * num_y * num_z, 0.f);

  // copy values from global data
  {
    Node whole_data_node;
    conduit::blueprint::mesh::examples::braid("hexs",
                                              data_size[0],
                                              data_size[1],
                                              data_size[2],
                                              whole_data_node);

    conduit::DataArray<double> whole_data_array = whole_data_node["fields/braid/values"].as_float64_array();

    // copy the subsection of data
    uint32_t offset = 0;
    uint32_t start = low[0] + low[1] * data_size[0] + low[2] * data_size[0] * data_size[1];
    for (uint32_t bz = 0; bz < num_z; ++bz) {
      for (uint32_t by = 0; by < num_y; ++by) {
        int data_idx = start + bz * data_size[0] * data_size[1] + by * data_size[0];
        for (uint32_t i = 0; i < num_x; ++i) {
          block_data[offset + i] = static_cast<FunctionType>(whole_data_array[data_idx + i]);
        }
        offset += num_x;
      }
    }
  }

  // build the local mesh.
  Node mesh;
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = num_x;
  mesh["coordsets/coords/dims/j"] = num_y;
  if (num_z > 1)    // if it's a 3D dataset
    mesh["coordsets/coords/dims/k"] = num_z;
  mesh["coordsets/coords/origin/x"] = low[0];
  mesh["coordsets/coords/origin/y"] = low[1];
  if (num_z > 1)    // if it's a 3D dataset
    mesh["coordsets/coords/origin/z"] = low[2];

  mesh["topologies/topo/type"] = "uniform";
  mesh["topologies/topo/coordset"] = "coords";
  mesh["fields/braids/association"] = "vertex";
  mesh["fields/braids/topology"] = "topo";
  mesh["fields/braids/values"].set_external(block_data);

  // assuming # of ranks == # of leaves
  int32_t task_id = mpi_rank;

  // publish
  a.publish(mesh);

  int32_t fanin = 2;
  FunctionType threshold = -FLT_MAX;
  std::vector<int64_t> in_ghosts({1, 1, 1, 1, 1, 1});

  // Make sure the output dir exists 
  string output_path = "";
  if (mpi_rank == 0)
  {
      output_path = prepare_output_dir();
  }
  else
  {
      output_path = output_dir();
  }
  string output_file = conduit::utils::join_file_path(output_path, "tout_babelflow_rel_mpi");

  // build filter Node
  Node pipelines;
  pipelines["pl1/f1/type"] = "bflow_pmt";
  pipelines["pl1/f1/params/field"] = "braids";
  pipelines["pl1/f1/params/fanin"] = int64_t(fanin);
  pipelines["pl1/f1/params/threshold"] = threshold;
  pipelines["pl1/f1/params/in_ghosts"].set_int64_vector(in_ghosts);
  pipelines["pl1/f1/params/gen_segment"] = int64_t(1);    // 1 -- means create a field with segmentation
  pipelines["pl1/f1/params/rel_field"] = int64_t(1);      // 1 -- means compute the relevance field

  std::vector<int64_t> radix_v{ mpi_size };
  switch (mpi_size)
  {
  case 2:
    radix_v = std::vector<int64_t>{2};
    break;

  case 4:
    radix_v = std::vector<int64_t>{2, 2};
    break;

  case 8:
    radix_v = std::vector<int64_t>{4, 2};
    break;

  case 16:
    radix_v = std::vector<int64_t>{4, 4};
    break;

  case 32:
    radix_v = std::vector<int64_t>{4, 8};
    break;

  case 64:
    radix_v = std::vector<int64_t>{8, 8};
    break;

  default:
    break;
  }  

  pipelines["pl1/f1/params/radices"].set_int64_vector(radix_v);

  ///
  Node extracts;
  extracts["e1/type"] = "relay";
  extracts["e1/pipeline"] = "pl1";
  extracts["e1/params/path"] = "seg";
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
  extracts["e1/params/fields"].append() = "segment";
  ///

  // Build scene node for volume rendering of the segmentation field
  double color[3] = {0.0, 0.0, 1.0};
  conduit::Node control_points;
  
  conduit::Node &point1 = control_points.append();
  point1["type"] = "rgb";
  point1["position"] = 0.;
  point1["color"].set_float64_ptr(color, 3);

  // conduit::Node &point2 = control_points.append();
  // point2["type"] = "rgb";
  // point2["position"] = 0.5;
  // color[0] = 1.0;
  // color[1] = 0.5;
  // color[2] = 0.25;
  // point2["color"].set_float64_ptr(color, 3);

  conduit::Node &point3 = control_points.append();
  point3["type"] = "rgb";
  point3["position"] = 1.0;
  color[0] = 1.0;
  color[1] = 0.2;
  color[2] = 0.2;
  point3["color"].set_float64_ptr(color, 3);

  conduit::Node &point4 = control_points.append();
  point4["type"] = "alpha";
  point4["position"] = 0.0;
  point4["alpha"] = 0.;

  conduit::Node &point5 = control_points.append();
  point5["type"] = "alpha";
  point5["position"] = 0.8;
  point5["alpha"] = 0.;
  
  conduit::Node &point6 = control_points.append();
  point6["type"] = "alpha";
  point6["position"] = 1.0;
  point6["alpha"] = 1.;
  
  Node scenes;

  scenes["s1/plots/p1/type"]  = "volume";
  scenes["s1/plots/p1/field"] = "segment";
  scenes["s1/plots/p1/color_table/control_points"] = control_points;
  
  scenes["s1/renders/r1/image_width"]  = 512;
  scenes["s1/renders/r1/image_height"] = 512;
  scenes["s1/renders/r1/image_prefix"] = output_file;
  scenes["s1/renders/r1/camera/azimuth"] = 30.0;
  scenes["s1/renders/r1/camera/elevation"] = 30.0;

  Node actions;

  Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;

  // Add the extracts
  Node &add_extracts = actions.append();
  add_extracts["action"] = "add_extracts";
  add_extracts["extracts"] = extracts;

  Node &add_scenes = actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  if (mpi_rank == 0)
    actions.print();

  start = clock();
  a.execute(actions);
  finish = clock();
  run_time = (static_cast<double>(finish) - static_cast<double>(start)) / CLOCKS_PER_SEC;
  MPI_Reduce(&run_time, &max_run_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    cout << dim << " " << max_run_time << endl;
  }

  a.close();

  MPI_Barrier(MPI_COMM_WORLD);

  // Check that we created an image
  EXPECT_TRUE(check_test_image(output_file, 0.1, "0"));
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;
    ::testing::InitGoogleTest(&argc, argv);
    int provided;
    auto err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(err == MPI_SUCCESS);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
