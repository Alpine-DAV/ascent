//
// Created by Li, Jixian on 2019-06-11.
//

#include <iomanip>
#include <iostream>
#include "b_pmt.h"

// CallBack Functions
static const uint8_t sPrefixSize = 4;
static const uint8_t sPostfixSize = sizeof(BabelFlow::TaskId) * 8 - sPrefixSize;
static const BabelFlow::TaskId sPrefixMask = ((1 << sPrefixSize) - 1) << sPostfixSize;

int local_compute(std::vector<BabelFlow::Payload> &inputs,
                  std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task)
{

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
         std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task)
{


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
                     std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task)
{

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
                  std::vector<BabelFlow::Payload> &output, BabelFlow::TaskId task)
{

  AugmentedMergeTree t;
  t.decode(inputs[0]);
  t.id(task & ~sPrefixMask);
  //t.writeToFile(task & ~sPrefixMask);
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


void ParallelMergeTree::Initialize()
{
  int my_rank;
  int mpi_size;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &mpi_size);

  BabelFlow::Payload payload = make_local_block(this->data_ptr, this->low, this->high, this->threshold);
  inputs[this->task_id] = payload;

  graph = KWayMerge(n_blocks, fanin);
  task_map = KWayTaskMap(mpi_size, &graph);
  MergeTree::setDimension(data_size);
  if (my_rank == 0) {
    FILE *fp = fopen("graph.dot", "w");
    graph.output_graph(mpi_size, &task_map, fp);
    fclose(fp);
  }
  master.initialize(graph, &task_map, MPI_COMM_WORLD, &c_map);
  master.registerCallback(1, local_compute);
  master.registerCallback(2, join);
  master.registerCallback(3, local_correction);
  master.registerCallback(4, write_results);
}

void ParallelMergeTree::Execute()
{
  master.run(inputs);
}

ParallelMergeTree::ParallelMergeTree(FunctionType *data_ptr, uint32_t task_id, const uint32_t *data_size,
                                     const uint32_t *n_blocks,
                                     const uint32_t *low, const uint32_t *high, uint32_t fanin,
                                     FunctionType threshold, MPI_Comm mpi_comm)
{
  this->data_ptr = data_ptr;
  this->task_id = task_id;
  this->data_size[0] = data_size[0];
  this->data_size[1] = data_size[1];
  this->data_size[2] = data_size[2];
  this->n_blocks[0] = n_blocks[0];
  this->n_blocks[1] = n_blocks[1];
  this->n_blocks[2] = n_blocks[2];
  this->low[0] = low[0];
  this->low[1] = low[1];
  this->low[2] = low[2];
  this->high[0] = high[0];
  this->high[1] = high[1];
  this->high[2] = high[2];
  this->fanin = fanin;
  this->threshold = threshold;
  this->comm = mpi_comm;
}
