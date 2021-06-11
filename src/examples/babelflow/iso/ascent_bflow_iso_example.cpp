//
// Created by Sergei Shudler on 2020-06-09.
//

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#define _NOMPI
#endif

#include <string.h> 
#include <iostream>
#include <cfloat>
#include <ctime>
#include <cassert>
#include <fstream>
#include <sstream>

#include <ascent.hpp>
#include <ascent_logging.hpp>
#include <conduit_blueprint.hpp>
#include <ascent_runtime_babelflow_comp_utils.hpp>


int main(int argc, char **argv)
{ 
  clock_t start_ts, finish_ts;

  if( argc < 6 ) 
  {
    std::cerr << argv[0] << " <dim> <fanin> <blk x> <blk y> <blk z> <radices>" << std::endl;
    exit(-1);
  }

  int provided;
  auto err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int arg_cnt = 1;
  int32_t dim = atoi(argv[arg_cnt++]);
  int32_t fanin = atoi(argv[arg_cnt++]);
  int32_t blk_x = atoi(argv[arg_cnt++]);
  int32_t blk_y = atoi(argv[arg_cnt++]);
  int32_t blk_z = atoi(argv[arg_cnt++]);

  std::vector<int64_t> radix_v;

  for( uint32_t i = arg_cnt; i < argc; ++i )
    radix_v.push_back(atoi(argv[i]));

  // user defines n_blocks per dimension
  // user provides the size of the whole data
  std::vector <int32_t> data_size({dim, dim, dim});
  std::vector <int32_t> n_blocks({blk_x, blk_y, blk_z});
  int32_t block_size[3] = {data_size[0] / n_blocks[0], data_size[1] / n_blocks[1], data_size[2] / n_blocks[2]};
  // compute the boundaries of the needed block
  std::vector <int32_t> low(3);
  std::vector <int32_t> high(3);
  low[0] = mpi_rank % n_blocks[0] * block_size[0];
  low[1] = mpi_rank / n_blocks[0] % n_blocks[1] * block_size[1];
  low[2] = mpi_rank / n_blocks[1] / n_blocks[2] % n_blocks[2] * block_size[2];
  high[0] = std::min(low[0] + block_size[0], data_size[0] - 1);
  high[1] = std::min(low[1] + block_size[1], data_size[1] - 1);
  high[2] = std::min(low[2] + block_size[2], data_size[2] - 1);

  // size of the local data
  int32_t num_x = high[0] - low[0] + 1;
  int32_t num_y = high[1] - low[1] + 1;
  int32_t num_z = high[2] - low[2] + 1;

  uint32_t block_data_size = num_x * num_y * num_z;
  std::vector<BabelFlow::FunctionType> block_data(block_data_size, 0.f);

  // if( mpi_rank == 0 )
  // {
  //   std::cout << "dim: " << dim << "  fanin: " << fanin << std::endl;
  //   std::cout << "n_blocks: " << n_blocks[0] << "  " << n_blocks[1] << "  " << n_blocks[2] << std::endl;
  //   std::cout << "block_size: " << block_size[0] << "  " << block_size[1] << "  " << block_size[2] << std::endl;
  //   std::cout << "high: " << high[0] << "  " << high[1] << "  " << high[2] << std::endl;
  //   std::cout << "low: " << low[0] << "  " << low[1] << "  " << low[2] << std::endl;
  // }

  conduit::Node whole_data_node;
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
        block_data[offset + i] = static_cast<BabelFlow::FunctionType>(whole_data_array[data_idx + i]);
      }
      offset += num_x;
    }
  }

  std::vector<double> isovals({1.0, 2.0, 3.0});

  // build the local mesh.
  conduit::Node mesh;
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

  ascent::Ascent a;
  conduit::Node ascent_opt;
  ascent_opt["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent_opt["runtime/type"] = "ascent";
  a.open(ascent_opt);

  // Publish
  a.publish(mesh);
  
  conduit::Node extract;
  extract["e1/type"] = "bflow_iso";
  extract["e1/params/field"] = "braids";
  extract["e1/params/iso_values"].set_float64_vector(isovals);
  extract["e1/params/image_name"] = "iso_img";
  extract["e1/params/radices"].set_int64_vector(radix_v);

  conduit::Node pipeline;
  pipeline["pl1/f1/type"] = "contour";
  pipeline["pl1/f1/params/field"] = "braids";
  pipeline["pl1/f1/params/iso_values"].set_float64_vector(isovals);

  // Extracts
  conduit::Node actions;
  conduit::Node &add_extract = actions.append();
  add_extract["action"] = "add_extracts";
  add_extract["extracts"] = extract;

  conduit::Node scenes;

  scenes["s1/plots/p1/type"]  = "pseudocolor";
  scenes["s1/plots/p1/field"] = "braids";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/renders/r1/image_width"]  = 1024;
  scenes["s1/renders/r1/image_height"] = 1024;
  scenes["s1/renders/r1/image_prefix"] = "img";
  // scenes["s1/renders/r1/camera/azimuth"] = 30.0;
  // scenes["s1/renders/r1/camera/elevation"] = 30.0;

  // conduit::Node actions;
  conduit::Node &add_pipeline = actions.append();
  add_pipeline["action"] = "add_pipelines";
  add_pipeline["pipelines"] = pipeline;

  conduit::Node &add_scenes = actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  // action.append()["action"] = "execute";

  // Print our full actions tree
  if (mpi_rank == 0)
  {
    std::cout << actions.to_yaml() << std::endl;
  }

  start_ts = clock();
  a.execute(actions);
  finish_ts = clock();

  double run_time, max_run_time;
  run_time = static_cast<double>(finish_ts - start_ts) / CLOCKS_PER_SEC;
  MPI_Reduce(&run_time, &max_run_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0)
    std::cout << "Exec time [s]: " << max_run_time << std::endl;

  a.close();
  MPI_Finalize();
  
  
  return 0;
}
