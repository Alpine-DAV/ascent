//
// Created by Li, Jixian on 2019-06-11.
//

#ifndef ASCENT_B_PMT_H
#define ASCENT_B_PMT_H

#include "BabelFlow/mpi/Controller.h"
#include "KWayMerge.h"
#include "KWayTaskMap.h"
#include "SortedUnionFindAlgorithm.h"
#include "SortedJoinAlgorithm.h"
#include "LocalCorrectionAlgorithm.h"
#include "MergeTree.h"
#include "AugmentedMergeTree.h"
#include "diy/mpi.hpp"


class ParallelMergeTree
{
public:
  ParallelMergeTree(FunctionType *data_ptr, uint32_t task_id, const uint32_t *data_size, const uint32_t *n_blocks,
                    const uint32_t *low, const uint32_t *high, uint32_t fanin,
                    FunctionType threshold, MPI_Comm mpi_comm);

  void Initialize();

  void Execute();

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

};


#endif //ASCENT_B_PMT_H
