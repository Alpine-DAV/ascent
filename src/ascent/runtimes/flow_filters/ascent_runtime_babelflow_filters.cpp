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
  t.computeSegmentation();
  t.writeToFileBinary(task & ~sPrefixMask);
  t.writeToFile(task & ~sPrefixMask);


  // Deleting input data
  for (int i = 0; i < inputs.size(); i++) {
    delete[] (char *) inputs[i].buffer();
  }
  inputs.clear();

  assert(output.size() == 0);
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
  //output = inputs;
  //return 1;
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
  i["output_port"] = "false";
}

#define IN_SCALAR

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
    
    // NOTE: when field is a vector the coords/spacing has dx/dy/dz
    int32_t dims[3] = {data_node["coordsets/coords/dims/i"].value(),data_node["coordsets/coords/dims/j"].value(),data_node["coordsets/coords/dims/k"].value()};

#ifdef IN_SCALAR
    int32_t spacing[3] = {data_node["coordsets/coords/spacing/x"].value(),data_node["coordsets/coords/spacing/y"].value(),data_node["coordsets/coords/spacing/z"].value()};
    int32_t origin[3] = {data_node["coordsets/coords/origin/x"].value(),data_node["coordsets/coords/origin/y"].value(),data_node["coordsets/coords/origin/z"].value()};
#else
    double spacing[3] = {data_node["coordsets/coords/spacing/dx"].value(),data_node["coordsets/coords/spacing/dy"].value(),data_node["coordsets/coords/spacing/dz"].value()};
    double origin[3] = {data_node["coordsets/coords/origin/x"].value(),data_node["coordsets/coords/origin/y"].value(),data_node["coordsets/coords/origin/z"].value()};
#endif

    int32_t low[3];
    int32_t high[3];

    int32_t global_low[3];
    int32_t global_high[3];
    int32_t data_size[3];

    int32_t n_blocks[3];
    std::cout<<"----------"<<rank<<"----------"<<std::endl;

    // TODO check dimensionality
    for(int i=0; i<3; i++){
      low[i] = origin[i]/spacing[i];
      high[i] = low[i] + dims[i];
      MPI_Allreduce(&low[i], &global_low[i], 1, MPI_INT, MPI_MIN, comm);
      MPI_Allreduce(&high[i], &global_high[i], 1, MPI_INT, MPI_MAX, comm);
      data_size[i] = global_high[i]-global_low[i];

      // normalize box
      low[i] -= global_low[i];
      high[i] = low[i] + dims[i] -1;

      n_blocks[i] = std::ceil(data_size[i]*1.0/dims[i]);
    }

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

    for(int i=0;i<3;i++){
      ParallelMergeTree::o_ghosts[i] = 2;
      ParallelMergeTree::n_ghosts[i] = 1;
    }
    //std::cout << p["field"].as_string() <<std::endl;

    //std::cout << "dtype " << data_node["fields/something/values"].dtype().print() <<std::endl;
    // get the data handle
#ifdef IN_SCALAR
    conduit::DataArray<double> array_mag = data_node["fields/mag/values"].as_float64_array();
#else
    conduit::DataArray<double> array_x = data_node["fields/something/values/x"].as_float64_array();
    conduit::DataArray<double> array_y = data_node["fields/something/values/y"].as_float64_array();
    conduit::DataArray<double> array_z = data_node["fields/something/values/z"].as_float64_array();
#endif
    //printf("NUMBER OF E %d\n", array_x.number_of_elements());

#ifndef IN_SCALAR
    FunctionType* array = new FunctionType[array_x.number_of_elements()]; 
    for(int i=0; i < array_x.number_of_elements(); i++)
      array[i] = std::sqrt(array_x[i]*array_x[i] + array_y[i]*array_y[i] + array_z[i]*array_z[i]);

    assert((dims[0]*dims[1]*dims[2]) == array_x.number_of_elements());
#else
    FunctionType* array = reinterpret_cast<FunctionType *>(array_mag.data_ptr());
#endif

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
    int32_t fanin = 2; //p["fanin"].as_int32();
    FunctionType threshold = -FLT_MAX;//0.0;//-999999.9; //p["threshold"].as_float();

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

    //delete [] array;

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
