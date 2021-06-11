
#include <ascent.hpp>
#include <ascent_logging.hpp>
#include <conduit_blueprint.hpp>

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

// #define BFLOW_PMT_DEBUG



typedef double FunctionType;

using namespace ascent;
using namespace conduit;

int main(int argc, char **argv)
{
  using namespace std;
  int provided;

  //int32_t dim = 256;
  //if (argc > 1) {
  //  dim = stoi(argv[1]);
  //}

  int32_t dim = 256;
  if (argc < 9) {
    fprintf(stderr,"Usage: %s -f <input_data> -d <Xdim> <Ydim> <Zdim> \
                    -p <dx> <dy> <dz> -m <fanin> -t <threshold>\n", argv[0]);
    return 0;
  }
  //arg parse
  int tot_blocks;
  int data_size_[3];             // {x_size, y_size, z_size}
  int block_decomp[3];     // block decomposition
  int min[3], max[3], size[3];  // block extents
  int nblocks;                  // my local number of blocks
  int ghost[6] = {0, 0, 0, 0, 0, 0};
  int share_face = 1;           // share a face among the blocks

  int test_block_size[3];
  int32_t valence = 2;
  //FunctionType threshold_ = (FunctionType)(-1)*FLT_MAX;
  FunctionType threshold_ = (int)(-1)*FLT_MAX;
  char* dataset;
  for (int i = 1; i < argc; i++){
    if (!strcmp(argv[i],"-d")){
      data_size_[0] = atoi(argv[++i]); 
      data_size_[1] = atoi(argv[++i]); 
      data_size_[2] = atoi(argv[++i]); 
    }
    if (!strcmp(argv[i],"-p")){
      block_decomp[0] = atoi(argv[++i]);
      block_decomp[1] = atoi(argv[++i]);
      block_decomp[2] = atoi(argv[++i]);
    }
    if (!strcmp(argv[i],"-m"))
      valence = atoi(argv[++i]);
    if (!strcmp(argv[i],"-t"))
      threshold_ = atof(argv[++i]);
    if (!strcmp(argv[i],"-f"))
      dataset = argv[++i];
  }
  dim =  block_decomp[0]*block_decomp[1]*block_decomp[2];

  auto err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(err == MPI_SUCCESS);

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
  vector <int32_t> data_size({data_size_[0], data_size_[1], data_size_[2]});
  vector <int32_t> n_blocks({ block_decomp[0],  block_decomp[1],  block_decomp[2]});
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

  // set the gloabl data
  vector<float> global_data(data_size[0]*data_size[1]*data_size[2], 0);
  {
    float mx = -FLT_MAX;
    float mn = FLT_MAX;
    ifstream rf(dataset, ios::out | ios::binary);
    if(!rf) {
      cout << "Cannot open file!" << endl;
      return 1;
    }

    for(int i = 0; i < data_size[0]*data_size[1]*data_size[2] ; i++)
    {
      rf.read( (char *)&global_data[i], sizeof(float));
      mx = std::max( mx, global_data[i] );
      mn = std::min( mn, global_data[i] );
    }

    rf.close();

    if( mpi_rank == 0 )
      std::cout << "Data range -- mx = " << mx << ", mn = " << mn << std::endl;
  }

  // size of the local data
  int32_t num_x = high[0] - low[0] + 1;
  int32_t num_y = high[1] - low[1] + 1;
  int32_t num_z = high[2] - low[2] + 1;
  vector<FunctionType> block_data(num_x * num_y * num_z, 0.f);

  // copy values from global data
  {
    // copy the subsection of data
    uint32_t offset = 0;
    uint32_t start = low[0] + low[1] * data_size[0] + low[2] * data_size[0] * data_size[1];
    for (uint32_t bz = 0; bz < num_z; ++bz) {
      for (uint32_t by = 0; by < num_y; ++by) {
        int data_idx = start + bz * data_size[0] * data_size[1] + by * data_size[0];
        for (uint32_t i = 0; i < num_x; ++i) {
          block_data[offset + i] = static_cast<FunctionType>(global_data[data_idx + i]);
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

  // output binary blocks for debugging purpose
  //  data<mpi_rank>.bin

#ifdef BFLOW_PMT_DEBUG
  {
    stringstream ss;
    ss << "data" << mpi_rank << ".bin";
    ofstream bofs(ss.str(), ios::out | ios::binary);
    bofs.write(reinterpret_cast<char *>(block_data.data()), block_data.size() * sizeof(FunctionType));
    bofs.close();
  }
#endif
  // output text block parameters
  //  data<mpi_rank>.params

#ifdef BFLOW_PMT_DEBUG
  {
    stringstream ss;
    ss << "data" << mpi_rank << ".params";
    ofstream ofs(ss.str());
    ofs << "dims/i = " << num_x << std::endl;
    ofs << "dims/j = " << num_y << std::endl;
    ofs << "dims/k = " << num_z << std::endl;
    ofs << "origin/x = " << low[0] << std::endl;
    ofs << "origin/y = " << low[1] << std::endl;
    ofs << "origin/z = " << low[2] << std::endl;
    ofs << "high[0] = " << high[0] << std::endl;
    ofs << "high[1] = " << high[1] << std::endl;
    ofs << "high[2] = " << high[2] << std::endl;
    ofs.flush();
    ofs.close();
  }
#endif

  // publish
  a.publish(mesh);
  
  vector<int64_t> in_ghosts({1, 1, 1, 1, 1, 1});
  
  // build pipeline Node for the filter
  Node pipelines;
  pipelines["pl1/f1/type"] = "bflow_pmt";
  pipelines["pl1/f1/params/field"] = "braids";
  pipelines["pl1/f1/params/fanin"] = int64_t(valence);
  pipelines["pl1/f1/params/threshold"] = threshold_;
  pipelines["pl1/f1/params/in_ghosts"].set_int64_vector(in_ghosts);
  pipelines["pl1/f1/params/gen_segment"] = int64_t(1);    // 1 -- means create a field with segmentation

  // Old parameters that we were using, now automatically computed by the filter
  //pipelines["pl1/f1/params/mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  //pipelines["pl1/f1/params/data_size"].set_int32_vector(data_size);
  //pipelines["pl1/f1/params/n_blocks"].set_int32_vector(n_blocks);
  //pipelines["pl1/f1/params/low"].set_int32_vector(low);
  //pipelines["pl1/f1/params/high"].set_int32_vector(high);
  //pipelines["pl1/f1/params/task_id"] = task_id;
  
  //  ### future work: supporting multiple blocks per node
  //  ### low and high should be defined in TaskId
  //  ### the following parameters should be defined in form of lists ###
  //  ### low, high, task_id
  //  ### we add additional number n_jobs to indicate number of blocks on this node ###
  //  ### low.size() = 3 * n_jobs
  //  ### high.size() = 3 * n_jobs
  //  ### task_id.size() = n_jobs
  //  ### data in form of <block_1, block_2, ..., block_n> size of each block is defined by high - low + 1 per dimension

  // extract to save to file the output
  Node extract;
  extract["e1/type"] = "relay";
  extract["e1/pipeline"] = "pl1";
  extract["e1/params/path"] = "seg";
  extract["e1/params/protocol"] = "blueprint/mesh/hdf5";
  extract["e1/params/fields"].append() = "segment";

  // pipelines

  Node action;
  Node &add_extract = action.append();
  add_extract["action"] = "add_extracts";
  add_extract["extracts"] = extract;

  action.append()["action"] = "execute";

  Node &add_pipelines = action.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;
  
  double color[3] = {0.0, 0.0, 1.0};
  conduit::Node control_points;
  
  conduit::Node &point1 = control_points.append();
  point1["type"] = "rgb";
  point1["position"] = 0.;
  point1["color"].set_float64_ptr(color, 3);

  conduit::Node &point2 = control_points.append();
  point2["type"] = "rgb";
  point2["position"] = 0.2;
  color[0] = 1.0;
  color[1] = 0.5;
  color[2] = 0.25;
  point2["color"].set_float64_ptr(color, 3);

  conduit::Node &point3 = control_points.append();
  point3["type"] = "rgb";
  point3["position"] = 1.0;
  color[0] = 1.0;
  color[1] = 0.0;
  color[2] = 1.0;
  point3["color"].set_float64_ptr(color, 3);

  conduit::Node &point4 = control_points.append();
  point4["type"] = "alpha";
  point4["position"] = 0.0;
  point4["alpha"] = 1.;

  conduit::Node &point5 = control_points.append();
  point5["type"] = "alpha";
  point5["position"] = 0.15;
  point5["alpha"] = 0.;
  
  conduit::Node &point6 = control_points.append();
  point6["type"] = "alpha";
  point6["position"] = 1.0;
  point6["alpha"] = 0.;
  
  Node& add_act2 = action.append();
  add_act2["action"] = "add_scenes";
  Node& scenes = add_act2["scenes"];

  // Do a volume rendering of the segmentation field
  scenes["s1/plots/p1/type"]  = "volume";
  scenes["s1/plots/p1/field"] = "segment";
  scenes["s1/plots/p1/color_table/control_points"] = control_points;
  
  scenes["s1/renders/r1/image_width"]  = 512;
  scenes["s1/renders/r1/image_height"] = 512;
  scenes["s1/renders/r1/image_name"]   = "segmentation";
  scenes["s1/renders/r1/camera/azimuth"] = 30.0;
  scenes["s1/renders/r1/camera/elevation"] = 30.0;
  
  // add a scene (s1) with one pseudocolor plot (p1) that
  // will render the result of our pipeline (pl1)
  //scenes["s1/plots/p1/type"] = "pseudocolor";
  //scenes["s1/plots/p1/pipeline"] = "pl1";
  //scenes["s1/plots/p1/field"] = "braids";
  //scenes["s1/image_name"] = "dataset";

  // render segmentation
  //scenes["s2/plots/p1/type"] = "pseudocolor";
  //scenes["s1/plots/p1/pipeline"] = "pl1";
  //scenes["s2/plots/p1/field"] = "segment";
  //scenes["s2/plots/p1/color_table/name"] = "Jet";
  //scenes["s2/image_name"] = "segmentation";

  // print our full actions tree
  if( mpi_rank == 0 )
    std::cout << action.to_yaml() << std::endl;

  action.append()["action"] = "execute";
  start = clock();
  a.execute(action);
  finish = clock();
  run_time = (static_cast<double>(finish) - static_cast<double>(start)) / CLOCKS_PER_SEC;
  MPI_Reduce(&run_time, &max_run_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0) 
  {
    cout << "Dim: " << dim << endl;
    cout << "Runtime(sec): " << max_run_time << endl;
  }

  a.close();
  MPI_Finalize();
  return 0;

}
