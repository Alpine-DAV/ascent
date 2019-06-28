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

//  using namespace std;
//  if (my_rank == 0) {
//    cout << "class data" << endl;
//    for (int i = 0; i < data_size[0] * data_size[1] * data_size[2]; ++i) {
//      cout << data[i] << " ";
//    }
//    cout << endl;
//  }

  // divide the data assign it to inputs
  uint32_t n_z = 0, n_y = 0, n_x = 0;
  for (uint32_t z = 0; z < data_size[2]; z += block_size[2]) {
    n_z++;
    for (uint32_t y = 0; y < data_size[1]; y += block_size[1]) {
      if (z == 0) n_y++;
      for (uint32_t x = 0; x < data_size[0]; x += block_size[0]) {
        if (y == 0 && z == 0) n_x++;
        uint32_t z_size = (data_size[2] <= z + block_size[2]) ? data_size[2] - z - 1 : block_size[2];
        uint32_t y_size = (data_size[1] <= y + block_size[1]) ? data_size[1] - y - 1 : block_size[1];
        uint32_t x_size = (data_size[0] <= x + block_size[0]) ? data_size[0] - x - 1 : block_size[0];

        uint32_t low[3] = {x, y, z};
        uint32_t high[3] = {x + x_size, y + y_size, z + z_size};
        DataBlock block(low, high);

        uint32_t num_x = high[0] - low[0] + 1;
        uint32_t num_y = high[1] - low[1] + 1;
        uint32_t num_z = high[2] - low[2] + 1;

        block.data = new FunctionType[num_x * num_y * num_z];
//        if (my_rank == 0 && x == 0 && y == 0 && z == 0) {
//          cout << "num_x " << num_x << endl;
//          cout << "num_y " << num_y << endl;
//          cout << "num_z " << num_z << endl;
//          cout << "block len " << num_x * num_y * num_z << endl;
//        }
        uint32_t offset = 0;
        uint32_t start = x + y * data_size[0] + z * data_size[0] * data_size[1];
        for (int bz = 0; bz < num_z; ++bz) {
          for (int by = 0; by < num_y; ++by) {
            FunctionType *data_ptr = this->data + start + bz * data_size[0] * data_size[1] + by * data_size[0];
            uint32_t data_len = num_x;
            memcpy(reinterpret_cast<char *>(block.data) + offset,
                   reinterpret_cast<char *>(data_ptr), data_len * sizeof(FunctionType));
            offset += data_len * sizeof(FunctionType);
          }
        }
        data_blocks.push_back(block);
      }
    }
  }

//  if (my_rank == 0) {
//    int block_n = 7;
//    cout << "check data block " << block_n << endl;
//    auto &&db = data_blocks[block_n];
//    cout << "low: " << db.low[0] << " " << db.low[1] << " " << db.low[2] << endl;
//    cout << "high: " << db.high[0] << " " << db.high[1] << " " << db.high[2] << endl;
//    cout << "data block " << block_n << " data" << endl;
//    auto db_len = (db.high[0] - db.low[0] + 1) * (db.high[1] - db.low[1] + 1) * (db.high[2] - db.low[2] + 1);
//    cout << "db len " << db_len << endl;
//    for (auto i = 0; i < db_len; ++i) {
//      cout << db.data[i] << " ";
//    }
//    cout << endl;
//
//  }

  std::vector<BabelFlow::TaskId> tasks;
  int factor = 1;
  while (data_blocks.size() / (factor * mpi_size)) {
    if (my_rank + ((factor - 1) * mpi_size) < data_blocks.size())
      tasks.push_back(my_rank + ((factor - 1) * mpi_size));
    factor++;
  }

  for (auto &&t: tasks) {
    BabelFlow::Payload local_block = make_local_block(data_blocks[t].data,
                                                      data_blocks[t].low,
                                                      data_blocks[t].high,
                                                      this->threshold);
    inputs[t] = local_block;
  }

//  for (int i = 0; i < data_blocks.size(); ++i) {
//    BabelFlow::Payload local_block = make_local_block(data_blocks[i].data,
//                                                      data_blocks[i].low,
//                                                      data_blocks[i].high,
//                                                      this->threshold);
////    if (my_rank == 0 && i == 7) {
////      cout << "input payload" << endl;
////      uint32_t b_size = sizeof(FunctionType *) + 6 * sizeof(GlobalIndexType) + sizeof(FunctionType);
////      char *msgbuffer = local_block.buffer();
////      FunctionType *dbuffer = reinterpret_cast<FunctionType *>(msgbuffer);
////      uint32_t data_len = (local_block.size() - b_size) / sizeof(FunctionType);
////      cout << "payload data len " << data_len << endl;
////      cout << "payload data" << endl;
////      for (int j = 0; j < data_len; ++j) {
////        cout << dbuffer[j] << " ";
////      }
////      cout << endl;
////    }
//    inputs[i] = local_block;
//  }

  for (auto &&db : data_blocks) delete[] db.data;

  uint32_t n_blocks[3] = {n_x, n_y, n_z};
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

ParallelMergeTree::ParallelMergeTree(FunctionType *data, int data_size[3], int block_size[3], int fanin,
                                     FunctionType threshold, MPI_Comm comm)
{
  char *bytes = new char[data_size[0] * data_size[1] * data_size[2] * sizeof(FunctionType)];
  memcpy(bytes, data, data_size[0] * data_size[1] * data_size[2] * sizeof(FunctionType));

  this->data = reinterpret_cast<FunctionType *>(bytes);
  this->data_size[0] = data_size[0];
  this->data_size[1] = data_size[1];
  this->data_size[2] = data_size[2];
  this->block_size[0] = block_size[0];
  this->block_size[1] = block_size[1];
  this->block_size[2] = block_size[2];
  this->fanin = fanin;
  this->threshold = threshold;
  this->comm = comm;
}
