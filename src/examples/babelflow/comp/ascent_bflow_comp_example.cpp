//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//
// Created by Sergei Shudler on 2020-06-09.
//


#include <mpi.h>

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
  if( argc < 8 ) 
  {
    std::cerr << "Usage: " << argv[0] << " -f <input_img> -d <width> <height> <channels> -m <fanin>" << std::endl;
    return 1;
  }
  
  int32_t width = 512, height = 512, channels = 3;
  int32_t fanin = 2;
  char* img_name = nullptr;
  std::vector<int64_t> radices({2, 2});
  
  // Parse args
  for( int i = 1; i < argc; i++ )
  {
    if( !strcmp(argv[i],"-d") )
    {
      width = atoi(argv[++i]); 
      height = atoi(argv[++i]); 
      channels = atoi(argv[++i]); 
    }
    
    if( !strcmp(argv[i],"-m") )
      fanin = atoi(argv[++i]);
      
    if( !strcmp(argv[i],"-f") )
      img_name = argv[++i];       // E.g., "raw_img_data_512_512_"
  }
  
  int provided;
  auto err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(err == MPI_SUCCESS);

  clock_t start, finish;
  double run_time, max_run_time;

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  ascent::Ascent a;
  conduit::Node ascent_opt;
  ascent_opt["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent_opt["runtime/type"] = "ascent";
  a.open(ascent_opt);
  
  // Image size includes pixel 'channels' and depth
  uint32_t buff_size = width*height*(channels + 1) + sizeof(uint32_t)*8;
  char* buff = new char[buff_size];
  std::stringstream img_name_ss;
  img_name_ss << img_name << mpi_rank << ".raw";
  std::ifstream img_ifs(img_name_ss.str(), std::ios::in | std::ios::binary);
  img_ifs.read(buff, buff_size);
  assert(img_ifs);
  
  uint32_t comp_image_channels = ascent::bflow_comp::ImageData::sNUM_CHANNELS;
  std::vector<unsigned char> pixels_vec( width*height*comp_image_channels, 255 );
  uint32_t min_channels = std::min((uint32_t)channels, comp_image_channels);
  // Convert channels
  for( uint32_t i = 0; i < width*height; ++i )
  {
    memcpy(pixels_vec.data() + i*comp_image_channels, buff + i*channels, min_channels);
  }
  std::vector<unsigned char> zbuf_vec((unsigned char*)(buff + width*height*channels), (unsigned char*)(buff + width*height*(channels+1)));
  
  // Build the dataset
  conduit::Node mesh;
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = width + 1;
  mesh["coordsets/coords/dims/j"] = height + 1;
  mesh["coordsets/coords/origin/x"] = 0;
  mesh["coordsets/coords/origin/y"] = 0;
  mesh["coordsets/coords/spacing/dx"] = 1;
  mesh["coordsets/coords/spacing/dy"] = 1;
  
  mesh["topologies/topo/type"] = "uniform";
  mesh["topologies/topo/coordset"] = "coords";
  
  mesh["fields/colors/association"] = "element";
  mesh["fields/colors/topology"] = "topo";
  mesh["fields/colors/values"].set_external(pixels_vec);
  
  mesh["fields/depth/association"] = "element";
  mesh["fields/depth/topology"] = "topo";
  mesh["fields/depth/values"].set_external(zbuf_vec);

  // Publish
  a.publish(mesh);
  
  conduit::Node extract;
  extract["e1/type"] = "bflow_comp";
  extract["e1/params/color_field"] = "colors";
  extract["e1/params/depth_field"] = "depth";
  extract["e1/params/image_name"] = "comp_img";
  extract["e1/params/fanin"] = int64_t(fanin);
  extract["e1/params/compositing"] = int64_t(2);   // 0 -- reduction, 1 -- bin-swap, 2 -- radix-k
  extract["e1/params/radices"].set_int64_vector(radices);

  // Pipelines
  conduit::Node action;
  conduit::Node &add_extract = action.append();
  add_extract["action"] = "add_extracts";
  add_extract["extracts"] = extract;

  action.append()["action"] = "execute";

  // Print our full actions tree
  std::cout << action.to_yaml() << std::endl;

  action.append()["action"] = "execute";
  start = clock();
  a.execute(action);
  finish = clock();
  run_time = (static_cast<double>(finish) - static_cast<double>(start)) / CLOCKS_PER_SEC;
  MPI_Reduce(&run_time, &max_run_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0)
    std::cout << "Exec time [s]: " << max_run_time << std::endl;

  a.close();
  MPI_Finalize();
  
  delete[] buff;
  
  return 0;
}
