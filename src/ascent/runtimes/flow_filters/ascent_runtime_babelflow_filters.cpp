//
// Created by Li, Jixian on 2019-06-04.
//

#include "ascent_runtime_babelflow_filters.hpp"

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <flow_workspace.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif

//-----------------------------------------------------------------------------
// -- begin ParallelMergeTree --
//-----------------------------------------------------------------------------

#include "BabelFlow/mpi/Controller.h"
#include "KWayMerge.h"
#include "KWayTaskMap.h"
#include "SortedUnionFindAlgorithm.h"
#include "SortedJoinAlgorithm.h"
#include "LocalCorrectionAlgorithm.h"
#include "MergeTree.h"
#include "AugmentedMergeTree.h"
#include "BabelFlow/PreProcessInputTaskGraph.hpp"
#include "BabelFlow/ModTaskMap.hpp"
#include "SimplificationGraph.h"
#include "SimpTaskMap.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>



class ParallelMergeTree {
public:
  ParallelMergeTree(FunctionType *data_ptr, int32_t task_id, const int32_t *data_size, const int32_t *n_blocks,
                    const int32_t *low, const int32_t *high, int32_t fanin,
                    FunctionType threshold, MPI_Comm mpi_comm);

  void Initialize();

  void Execute();

  void ExtractSegmentation(FunctionType* output_data_ptr);

  static int DownSizeGhosts(std::vector<BabelFlow::Payload> &inputs, std::vector<BabelFlow::Payload> &output,
                            BabelFlow::TaskId task);

  static uint32_t o_ghosts[6];
  static uint32_t n_ghosts[6];
  static uint32_t s_data_size[3];

private:
  FunctionType *data_ptr;
  uint32_t task_id;
  uint32_t data_size[3];
  uint32_t n_blocks[3];
  uint32_t low[3];
  uint32_t high[3];
  uint32_t fanin;

  FunctionType threshold;
  std::map<BabelFlow::TaskId, BabelFlow::Payload> inputs;

  MPI_Comm comm;
  BabelFlow::mpi::Controller master;
  BabelFlow::ControllerMap c_map;
  KWayTaskMap task_map;
  KWayMerge graph;

  BabelFlow::PreProcessInputTaskGraph<KWayMerge> modGraph;
  BabelFlow::ModTaskMap<KWayTaskMap> modMap;

};


// CallBack Functions
static const uint8_t sPrefixSize = 4;
static const uint8_t sPostfixSize = sizeof(BabelFlow::TaskId) * 8 - sPrefixSize;
static const BabelFlow::TaskId sPrefixMask = ((1 << sPrefixSize) - 1) << sPostfixSize;
uint32_t ParallelMergeTree::o_ghosts[];
uint32_t ParallelMergeTree::n_ghosts[];
uint32_t ParallelMergeTree::s_data_size[];

// Static ptr for local data to be passed to computeSegmentation()
FunctionType* sLocalData = NULL;


int local_compute(std::vector<BabelFlow::Payload> &inputs,
                  std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  sorted_union_find_algorithm(inputs, output, task);
  /*
  MergeTree t;

  //fprintf(stderr,"LOCAL COMPUTE performed by task %d\n", task);
  t.decode(output[0]);

  t.writeToFile(task);
*/
  // Deleting input data
  for (int i = 0; i < inputs.size(); i++) {
    delete[] (char *) inputs[i].buffer();
  }
  inputs.clear();

  return 1;
}


int join(std::vector<BabelFlow::Payload> &inputs,
         std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {


  //fprintf(stderr, "Task : %d : Started with join algorithm\n", task);
  sorted_join_algorithm(inputs, output, task);
  //fprintf(stderr, "Task : %d : Done with join algorithm\n", task);
/*
  MergeTree join_tree;

  join_tree.decode(output[0]);
  join_tree.writeToFile(task+1000);
  */
  // Deleting input data
  for (int i = 0; i < inputs.size(); i++) {
    delete[] (char *) inputs[i].buffer();
  }
  inputs.clear();

  return 0;
}

int local_correction(std::vector<BabelFlow::Payload> &inputs,
                     std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  //if ((task & ~sPrefixMask) == 237)
  local_correction_algorithm(inputs, output, task);

  // Deleting input data
  for (int i = 0; i < inputs.size(); i++) {
    delete[] (char *) inputs[i].buffer();
  }
  inputs.clear();

  //fprintf(stderr,"CORRECTION performed by task %d\n", task & ~sPrefixMask);
  return 1;
}

int write_results(std::vector<BabelFlow::Payload> &inputs,
                  std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  AugmentedMergeTree t;
  t.decode(inputs[0]);
  t.id(task & ~sPrefixMask);
  //t.writeToFile(task & ~sPrefixMask);
  t.persistenceSimplification(1.f);
  t.computeSegmentation(sLocalData);
  t.writeToFileBinary(task & ~sPrefixMask);
  t.writeToFile(task & ~sPrefixMask);
  //t.writeToHtmlFile(task & ~sPrefixMask);

  // Set the final tree as an output so that it could be extracted later
  output[0] = t.encode();

  // Deleting input data
  for (int i = 0; i < inputs.size(); i++) {
    delete[] (char *) inputs[i].buffer();
  }
  inputs.clear();

  //fprintf(stderr,"WRITING RESULTS performed by %d\n", task & ~sPrefixMask);
  return 1;
}


int ParallelMergeTree::DownSizeGhosts(std::vector<BabelFlow::Payload> &inputs, std::vector<BabelFlow::Payload> &output,
                                      BabelFlow::TaskId task) {
  FunctionType *block_data;
  FunctionType threshold;
  GlobalIndexType low[3];
  GlobalIndexType high[3];

  decode_local_block(inputs[0], &block_data, low, high, threshold);

  GlobalIndexType xsize = high[0] - low[0] + 1;
  GlobalIndexType ysize = high[1] - low[1] + 1;
  GlobalIndexType zsize = high[2] - low[2] + 1;

  int dnx = (low[0] == s_data_size[0]) ? 0 : o_ghosts[0] - n_ghosts[0];
  int dny = (low[1] == s_data_size[1]) ? 0 : o_ghosts[1] - n_ghosts[1];
  int dnz = (low[2] == s_data_size[2]) ? 0 : o_ghosts[2] - n_ghosts[2];
  int dpx = (high[0] == s_data_size[0] - 1) ? 0 : o_ghosts[3] - n_ghosts[3];
  int dpy = (high[1] == s_data_size[1] - 1) ? 0 : o_ghosts[4] - n_ghosts[4];
  int dpz = (high[2] == s_data_size[2] - 1) ? 0 : o_ghosts[5] - n_ghosts[5];

  int numx = xsize - dnx - dpx;
  int numy = ysize - dny - dpy;
  int numz = zsize - dnz - dpz;

  char *n_block_data = new char[numx * numy * numz * sizeof(FunctionType)];
  size_t offset = 0;
  for (int z = 0; z < zsize; ++z) {
    if (z >= dnz && z < zsize - dpz) {
      for (int y = 0; y < ysize; ++y) {
        if (y >= dny && y < ysize - dpy) {
          FunctionType *data_ptr = block_data + y * xsize + z * ysize * xsize;
          memcpy(n_block_data + offset, (char *) data_ptr, numx * sizeof(FunctionType));
          offset += numx * sizeof(FunctionType);
        }
      }
    }
  }

  GlobalIndexType n_low[3];
  GlobalIndexType n_high[3];
  n_low[0] = low[0] + dnx;
  n_low[1] = low[1] + dny;
  n_low[2] = low[2] + dnz;

  n_high[0] = high[0] - dpx;
  n_high[1] = high[1] - dpy;
  n_high[2] = high[2] - dpz;

  output.resize(1);
  output[0] = make_local_block((FunctionType*)n_block_data, n_low, n_high, threshold);

  delete[] inputs[0].buffer();
  return 0;
}

int pre_proc(std::vector<BabelFlow::Payload> &inputs,
             std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task) {

  return ParallelMergeTree::DownSizeGhosts(inputs, output, task);
  //printf("this is where the preprocessing supposed to happend for Task %d\n", task);
  // output = inputs;
  // return 1;
}


void ParallelMergeTree::Initialize() {
  int my_rank = 0;
  int mpi_size = 1;

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &mpi_size);
#endif

  graph = KWayMerge(n_blocks, fanin);
  task_map = KWayTaskMap(mpi_size, &graph);

  // SimplificationGraph g(&graph, &task_map, mpi_size);
  // SimpTaskMap m(&task_map);
  // m.update(g);
  // if (my_rank == 0){
  //   FILE *fp = fopen("simpgraph.dot", "w");
  //   g.output_graph(mpi_size, &m, fp);
  //   fclose(fp);
  // }

  modGraph = BabelFlow::PreProcessInputTaskGraph<KWayMerge>(mpi_size, &graph, &task_map);
  modMap = BabelFlow::ModTaskMap<KWayTaskMap>(&task_map);
  modMap.update(modGraph);

  MergeTree::setDimension(data_size);
  // if (my_rank == 0) {
  //   FILE *ofp = fopen("original_graph.dot", "w");
  //   graph.output_graph(mpi_size, &task_map, ofp);
  //   fclose(ofp);
  //   FILE *fp = fopen("graph.dot", "w");
  //   modGraph.output_graph(mpi_size, &modMap, fp);
  //   fclose(fp);
  // }

  master.initialize(modGraph, &modMap, MPI_COMM_WORLD, &c_map);
  master.registerCallback(1, local_compute);
  master.registerCallback(2, join);
  master.registerCallback(3, local_correction);
  master.registerCallback(4, write_results);
  master.registerCallback(modGraph.newCallBackId, pre_proc);

  BabelFlow::Payload payload = make_local_block(this->data_ptr, this->low, this->high, this->threshold);
  inputs[modGraph.new_tids[this->task_id]] = payload;
}

void ParallelMergeTree::Execute() {
  master.run(inputs);
}

void ParallelMergeTree::ExtractSegmentation(FunctionType* output_data_ptr) {

  // Get the outputs map (maps task IDs to outputs) from the controller
  std::map<BabelFlow::TaskId,std::vector<BabelFlow::Payload> >& outputs = master.getAllOutputs();

  for (auto iter = outputs.begin(); iter != outputs.end(); ++iter) {
    // Output task should be only 'write_results' task
    assert(graph.task(graph.gId(iter->first)).callback() == 4);

    AugmentedMergeTree t;
    t.decode((iter->second)[0]);    // only one output per task

    // Copy the segmentation labels into the provided data array -- assume it
    // has enough space (sampleCount() -- the local data size)
    for (uint32_t i = 0; i < t.sampleCount(); ++i)
      output_data_ptr[i] = (float)t.label(i);
  }
}


ParallelMergeTree::ParallelMergeTree(FunctionType *data_ptr, int32_t task_id, const int32_t *data_size,
                                     const int32_t *n_blocks,
                                     const int32_t *low, const int32_t *high, int32_t fanin,
                                     FunctionType threshold, MPI_Comm mpi_comm) {
  this->data_ptr = data_ptr;
  this->task_id = static_cast<uint32_t>(task_id);
  this->data_size[0] = static_cast<uint32_t>(data_size[0]);
  this->data_size[1] = static_cast<uint32_t>(data_size[1]);
  this->data_size[2] = static_cast<uint32_t>(data_size[2]);
  this->n_blocks[0] = static_cast<uint32_t>(n_blocks[0]);
  this->n_blocks[1] = static_cast<uint32_t>(n_blocks[1]);
  this->n_blocks[2] = static_cast<uint32_t>(n_blocks[2]);
  this->low[0] = static_cast<uint32_t>(low[0]);
  this->low[1] = static_cast<uint32_t>(low[1]);
  this->low[2] = static_cast<uint32_t>(low[2]);
  this->high[0] = static_cast<uint32_t>(high[0]);
  this->high[1] = static_cast<uint32_t>(high[1]);
  this->high[2] = static_cast<uint32_t>(high[2]);
  this->fanin = static_cast<uint32_t>(fanin);
  this->threshold = threshold;
  this->comm = mpi_comm;

  // Store the local data as a static pointer so that it could be passed later to
  // computeSegmentation function -- probably a better solution is needed
  sLocalData = data_ptr;
}


//-----------------------------------------------------------------------------
// -- end ParallelMergeTree --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
///
/// BabelFlow Filter
///
//-----------------------------------------------------------------------------

void ascent::runtime::filters::BabelFlow::declare_interface(conduit::Node &i) {
  i["type_name"] = "babelflow";
  i["port_names"].append() = "in";
  i["output_port"] = "true";  // true -- means filter, false -- means extract
}

//#define INPUT_SCALAR

void ascent::runtime::filters::BabelFlow::execute() {

  // DEBUG prints
#if 0
  {
    auto in = input<DataObject>(0)->as_node();
    auto itr_dnode = in->children();
    while(itr_dnode.has_next())
    {
      auto& data_node = itr_dnode.next();
      std::string cld_dname = data_node.name();
      std::cout << "dnode name " <<cld_dname  << std::endl; //<< ": " << cld.to_json()

      conduit::NodeIterator itr = data_node["fields/"].children();
      while(itr.has_next())
      {
            conduit::Node &cld = itr.next();
            std::string cld_name = itr.name();
            std::cout << "\tname " <<cld_name  << std::endl; //<< ": " << cld.to_json()
      }
    }

  }

#endif

  if (op == PMT) {
    // connect to the input port and get the parameters
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("BabelFlow filter requires a DataObject");
    }

    DataObject *d_input = input<DataObject>(0);
    std::shared_ptr<conduit::Node> n_input = d_input->as_node();

    conduit::Node p = params();
    auto *in = n_input.get();
    
    auto &data_node = in->children().next();

    // check if coordset uniform
    if(data_node.has_path("coordsets/coords/type"))
    {
      std::string coordSetType = data_node["coordsets/coords/type"].as_string();
      if (coordSetType != "uniform")
      {
          // error
          ASCENT_ERROR("BabelFlow filter currenlty only works with uniform grids");
      }
    }
    else
      ASCENT_ERROR("BabelFlow filter could not find coordsets/coords/type");

    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    MPI_Comm_rank(comm, &rank);
#endif

    //data_node["coordsets/coords"].print();
    //data_node["topologies"].print();

    const int ndims = data_node.has_path("coordsets/coords/dims/k") ? 3 : 2;

    // NOTE: when field is a vector the coords/spacing has dx/dy/dz
    int32_t dims[ndims];
    dims[0] = data_node["coordsets/coords/dims/i"].value();
    dims[1] = data_node["coordsets/coords/dims/j"].value();
    if(ndims > 2)
      dims[2] = data_node["coordsets/coords/dims/k"].value();

// #ifdef INPUT_SCALAR
    int32_t spacing[ndims];
    for(int i=0; i < ndims; i++) spacing[i] = 1;

    if(data_node.has_path("coordsets/coords/spacing")){
      spacing[0] = data_node["coordsets/coords/spacing/x"].value();
      spacing[1] = data_node["coordsets/coords/spacing/y"].value();
      if(ndims > 2)
        spacing[2] = data_node["coordsets/coords/spacing/z"].value();
    }

    int32_t origin[ndims];
    origin[0] = data_node["coordsets/coords/origin/x"].value();
    origin[1] = data_node["coordsets/coords/origin/y"].value();
    
    if(ndims > 2)
      origin[2] = data_node["coordsets/coords/origin/z"].value();
// #else
//     double spacing[ndims] = {data_node["coordsets/coords/spacing/dx"].value(),data_node["coordsets/coords/spacing/dy"].value(),data_node["coordsets/coords/spacing/dz"].value()};
//     double origin[ndims] = {data_node["coordsets/coords/origin/x"].value(),data_node["coordsets/coords/origin/y"].value(),data_node["coordsets/coords/origin/z"].value()};
// #endif

    // Inputs of PMT assume 3D dataset
    int32_t low[3] = {0,0,0};
    int32_t high[3] = {1,1,1};

    int32_t global_low[3] = {0,0,0};
    int32_t global_high[3] = {1,1,1};
    int32_t data_size[3] = {1,1,1};

    int32_t n_blocks[3] = {1,1,1};

    for(int i=0;i<ndims;i++){
      if(p.has_path("in_ghosts"))
        ParallelMergeTree::o_ghosts[i] = p["in_ghosts"].as_int64();
      else
        ParallelMergeTree::o_ghosts[i] = 1;

      ParallelMergeTree::n_ghosts[i] = 1;
    }

    for(int i=0; i<ndims; i++){
      low[i] = origin[i]/spacing[i];
      high[i] = low[i] + dims[i] -1;

      MPI_Allreduce(&low[i], &global_low[i], 1, MPI_INT, MPI_MIN, comm);
      MPI_Allreduce(&high[i], &global_high[i], 1, MPI_INT, MPI_MAX, comm);
      data_size[i] = global_high[i]-global_low[i]+1;

      // normalize box
      low[i] -= global_low[i];
      high[i] = low[i] + dims[i] -1;

      n_blocks[i] = std::ceil(data_size[i]*1.0/dims[i]);
    }

    // int32_t dn[3], dp[3];
    // dn[0] = (low[0] == data_size[0]) ? 0 : ParallelMergeTree::o_ghosts[0] - ParallelMergeTree::n_ghosts[0];
    // dn[1] = (low[1] == data_size[1]) ? 0 : ParallelMergeTree::o_ghosts[1] - ParallelMergeTree::n_ghosts[1];
    // dn[2] = (low[2] == data_size[2]) ? 0 : ParallelMergeTree::o_ghosts[2] - ParallelMergeTree::n_ghosts[2];
    // dp[0] = (high[0] == data_size[0] - 1) ? 0 : ParallelMergeTree::o_ghosts[3] - ParallelMergeTree::n_ghosts[3];
    // dp[1] = (high[1] == data_size[1] - 1) ? 0 : ParallelMergeTree::o_ghosts[4] - ParallelMergeTree::n_ghosts[4];
    // dp[2] = (high[2] == data_size[2] - 1) ? 0 : ParallelMergeTree::o_ghosts[5] - ParallelMergeTree::n_ghosts[5];

    // for(int i=0; i<ndims; i++){
    //   low[i] += dn[i];
    //   high[i] -= dp[i];
    // }
    

#if 0
    std::cout<<"----------"<<rank<<"----------"<<std::endl;

    // Reduce all of the local sums into the global sum
    
    std::cout << "dims " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
    std::cout << "low " << low[0] << " " << low[1] << " " << low[2] << std::endl;
    std::cout << "high " << high[0] << " " << high[1] << " " << high[2] << std::endl;

    if(rank==0){
      std::cout << "*data_size " << data_size[0] << " " << data_size[1] << " " << data_size[2] << std::endl;
      std::cout << "*global_low " << global_low[0] << " " << global_low[1] << " " << global_low[2] << std::endl;
      std::cout << "*global_high " << global_high[0] << " " << global_high[1] << " " << global_high[2] << std::endl;
      std::cout << "*n_blocks " << n_blocks[0] << " " << n_blocks[1] << " " << n_blocks[2] << std::endl;
    }

    //data_node["fields/"].print();
    std::cout<<"----------------------"<<std::endl;
#endif

    //std::cout << p["field"].as_string() <<std::endl;

    //std::cout << "dtype " << data_node["fields/something/values"].dtype().print() <<std::endl;
    // get the data handle

    conduit::Node& fields_root_node = data_node["fields"];
    conduit::Node& field_node = fields_root_node[p["field"].as_string()];

// #ifdef INPUT_SCALAR
    conduit::DataArray<double> array_mag = field_node["values"].as_float64_array();
// #else
//     conduit::DataArray<double> array_x = data_node["fields/something/values/x"].as_float64_array();
//     conduit::DataArray<double> array_y = data_node["fields/something/values/y"].as_float64_array();
//     conduit::DataArray<double> array_z = data_node["fields/something/values/z"].as_float64_array();
// #endif
    //printf("NUMBER OF E %d\n", array_x.number_of_elements());

// #ifndef INPUT_SCALAR
//     FunctionType* array = new FunctionType[array_x.number_of_elements()]; 
//     for(int i=0; i < array_x.number_of_elements(); i++)
//       array[i] = std::sqrt(array_x[i]*array_x[i] + array_y[i]*array_y[i] + array_z[i]*array_z[i]);

//     //assert((dims[0]*dims[1]*dims[2]) == array_x.number_of_elements());
// #else
    FunctionType* array = reinterpret_cast<FunctionType *>(array_mag.data_ptr());
//#endif

    //conduit::DataArray<float> array = data_node[p["data_path"].as_string()].as_float32_array();

#if 0
    std::stringstream ss;
    ss << "block_" << dims[0] << "_" << dims[1] << "_" << dims[2] <<"_low_"<< low[0] << "_"<< low[1] << "_"<< low[2] << ".raw";
    std::fstream fil;
    fil.open(ss.str().c_str(), std::ios::out | std::ios::binary);
    fil.write(reinterpret_cast<char *>(array), (dims[0]*dims[1]*dims[2])*sizeof(FunctionType));
    fil.close();

    MPI_Barrier(comm);
#endif

    // int32_t *data_size = p["data_size"].as_int32_ptr();
    // int32_t *low = p["low"].as_int32_ptr();
    // int32_t *high = p["high"].as_int32_ptr();
    // int32_t *n_blocks = p["n_blocks"].as_int32_ptr();
    int32_t task_id = rank;
    // if (!p.has_child("task_id") || p["task_id"].as_int32() == -1) {
    //   MPI_Comm_rank(comm, &task_id);
    // } else {
    //   task_id = p["task_id"].as_int32();
    // }
    int64_t fanin = p["fanin"].as_int64();
    FunctionType threshold = p["threshold"].as_float64();
    int64_t gen_field = p["gen_segment"].as_int64();

    // create ParallelMergeTree instance and run
    ParallelMergeTree pmt(array, 
                          task_id,
                          data_size,
                          n_blocks,
                          low, high,
                          fanin, threshold, comm);

    ParallelMergeTree::s_data_size[0] = data_size[0];
    ParallelMergeTree::s_data_size[1] = data_size[1];
    ParallelMergeTree::s_data_size[2] = data_size[2];

    pmt.Initialize();
    pmt.Execute();

    if (gen_field) {
      // Generate new field 'segment'
      data_node["fields/segment/association"] = field_node["association"].as_string();
      data_node["fields/segment/topology"] = field_node["topology"].as_string();

      // New field data
      int32_t num_x = high[0] - low[0] + 1;
      int32_t num_y = high[1] - low[1] + 1;
      int32_t num_z = high[2] - low[2] + 1;
      std::vector<FunctionType> seg_data(num_x*num_y*num_z, 0.f);

      pmt.ExtractSegmentation(seg_data.data());

      data_node["fields/segment/values"].set(seg_data);

      // DEBUG -- write raw segment data to disk
      // {
      //   std::stringstream ss;
      //   ss << "segment_data_" << rank << ".bin";
      //   std::ofstream bofs(ss.str(), std::ios::out | std::ios::binary);
      //   bofs.write(reinterpret_cast<char *>(reldata.data()), num_x*num_y*num_z * sizeof(float));
      //   bofs.close();
      // }

      // DEBUG -- verify modified BP node with 'segment' field
      // conduit::Node info;
      // if (conduit::blueprint::verify("mesh", *in, info))
      //   std::cout << "BP with new field verify -- successful" << std::endl;
      // else
      //   std::cout << "BP with new field verify -- failed" << std::endl;
      
      d_input->reset_vtkh_collection();
    }
    
    set_output<DataObject>(d_input);
  } else {
    return;
  }
}

bool ascent::runtime::filters::BabelFlow::verify_params(const conduit::Node &params, conduit::Node &info) {
  if (params.has_child("task")) {
    std::string task_str(params["task"].as_string());
    if (task_str == "pmt") {
      this->op = PMT;
    } else {
      std::cerr << "[Error] ascent::BabelFlow\nUnknown task \"" << task_str << "\"" << std::endl;
      return false;
    }
  } else {
    std::cerr << "[Error] ascent::BabelFlow\ntask need to be specified" << std::endl;
    return false;
  }
  return true;
}
