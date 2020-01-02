//
// Created by Li, Jixian on 2019-06-11.
//

#include <iomanip>
#include <iostream>
#include "b_pmt.hpp"

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
  printf("this is where the preprocessing supposed to happend for Task %d\n", task);
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


