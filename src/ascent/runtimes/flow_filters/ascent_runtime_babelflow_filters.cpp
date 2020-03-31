//
// Created by Li, Jixian on 2019-06-04.
//

#include "ascent_runtime_babelflow_filters.hpp"

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>

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
  //printf("this is where the preprocessing supposed to happend for Task %d\n", task);
  output = inputs;
  return 1;
}


void ParallelMergeTree::Initialize() {
  int my_rank;
  int mpi_size;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &mpi_size);

  graph = KWayMerge(n_blocks, fanin);
  task_map = KWayTaskMap(mpi_size, &graph);

  SimplificationGraph g(&graph, &task_map, mpi_size);
  SimpTaskMap m(&task_map);
  m.update(g);
  if (my_rank == 0){
    FILE *fp = fopen("simpgraph.dot", "w");
    g.output_graph(mpi_size, &m, fp);
    fclose(fp);
  }


  modGraph = BabelFlow::PreProcessInputTaskGraph<KWayMerge>(mpi_size, &graph, &task_map);
  modMap = BabelFlow::ModTaskMap<KWayTaskMap>(&task_map);
  modMap.update(modGraph);

  MergeTree::setDimension(data_size);
  if (my_rank == 0) {
    FILE *ofp = fopen("original_graph.dot", "w");
    graph.output_graph(mpi_size, &task_map, ofp);
    fclose(ofp);
    FILE *fp = fopen("graph.dot", "w");
    modGraph.output_graph(mpi_size, &modMap, fp);
    fclose(fp);
  }

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


void ascent::runtime::filters::BabelFlow::execute() {
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

    // get the data handle
    conduit::Node& field_node = data_node[p["field_path"].as_string()];
    conduit::DataArray<float> array = field_node["values"].as_float32_array();

    // get the parameters
    MPI_Comm comm = MPI_Comm_f2c(p["mpi_comm"].as_int());
    int rank;
    MPI_Comm_rank(comm, &rank);
    int32_t *data_size = p["data_size"].as_int32_ptr();
    int32_t *low = p["low"].as_int32_ptr();
    int32_t *high = p["high"].as_int32_ptr();
    int32_t *n_blocks = p["n_blocks"].as_int32_ptr();
    int32_t task_id;
    if (!p.has_child("task_id") || p["task_id"].as_int32() == -1) {
      MPI_Comm_rank(comm, &task_id);
    } else {
      task_id = p["task_id"].as_int32();
    }
    int32_t fanin = p["fanin"].as_int32();
    FunctionType threshold = p["threshold"].as_float();
    int32_t gen_field = p["gen_segment"].as_int32();

    // create ParallelMergeTree instance and run
    ParallelMergeTree pmt(reinterpret_cast<FunctionType *>(array.data_ptr()),
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
