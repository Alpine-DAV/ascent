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
#include "BabelFlow/PreProcessInputTaskGraph.hpp"
#include "BabelFlow/ModTaskMap.hpp"
#include "SimplificationGraph.h"
#include "SimpTaskMap.h"


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

#endif //ASCENT_B_PMT_H
