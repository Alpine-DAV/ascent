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

struct DataBlock
{
  uint32_t low[3];
  uint32_t high[3];
  FunctionType *data;

  DataBlock(uint32_t l[3], uint32_t h[3])
  {
    low[0] = l[0];
    low[1] = l[1];
    low[2] = l[2];
    high[0] = h[0];
    high[1] = h[1];
    high[2] = h[2];
    data = nullptr;
  }

//  ~DataBlock()
//  {
//    delete[] data;
//  }
};

class ParallelMergeTree
{
public:
  ParallelMergeTree(FunctionType *data, int data_size[3], int block_size[3], int fanin, FunctionType threshold,
                    MPI_Comm comm);

  void Initialize();

  void Execute();

private:
  std::vector<DataBlock> data_blocks;
  std::map<BabelFlow::TaskId, BabelFlow::Payload> inputs;
  GlobalIndexType data_size[3];
  uint32_t block_size[3];
  uint32_t fanin;
  FunctionType threshold;
  MPI_Comm comm;

  FunctionType *data;

  BabelFlow::mpi::Controller master;
  BabelFlow::ControllerMap c_map;
  KWayTaskMap task_map;
  KWayMerge graph;

};


#endif //ASCENT_B_PMT_H
