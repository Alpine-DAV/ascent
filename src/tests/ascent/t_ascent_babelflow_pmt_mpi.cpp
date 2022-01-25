//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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

TEST(ascent_babelfow_pmt_mpi, test_babelfow_pmt_mpi)
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
  string output_file = conduit::utils::join_file_path(output_path, "tout_babelflow_pmt_mpi");

  // build filter Node
  Node pipelines;
  pipelines["pl1/f1/type"] = "bflow_pmt";
  pipelines["pl1/f1/params/field"] = "braids";
  pipelines["pl1/f1/params/fanin"] = int64_t(fanin);
  pipelines["pl1/f1/params/threshold"] = threshold;
  pipelines["pl1/f1/params/in_ghosts"].set_int64_vector(in_ghosts);
  pipelines["pl1/f1/params/gen_segment"] = int64_t(1);    // 1 -- means create a field with segmentation

  ///
  Node extracts;
  extracts["e1/type"] = "relay";
  extracts["e1/pipeline"] = "pl1";
  extracts["e1/params/path"] = "seg";
  extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
  extracts["e1/params/fields"].append() = "segment";
  ///

  // Build scene node for volume rendering of the segmentation field
  Node scenes;

  scenes["s1/plots/p1/type"]  = "volume";
  scenes["s1/plots/p1/field"] = "segment";
  // scenes["s1/plots/p1/color_table/control_points"] = control_points;
  
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
