//
// Created by Li, Jixian on 2019-06-04.
//

#ifndef ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H
#define ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H

#include <flow_filter.hpp>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <babelflow/pmt/b_pmt.h>

namespace ascent
{
namespace runtime
{
namespace filter
{

enum op
{
  NOOP = 0,
  PMT
};

class BabelFlow : public ::flow::Filter
{
private:
  op op = NOOP;
  int mpi_size;
  int mpi_rank;
  std::vector<int> mr_data;

public:
  BabelFlow():mpi_size(0),mpi_rank(0){};
  void declare_interface(conduit::Node &i) override;

  void execute() override;

  bool verify_params(const conduit::Node &params, conduit::Node &info) override;
};
}
}
}


#endif //ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H
