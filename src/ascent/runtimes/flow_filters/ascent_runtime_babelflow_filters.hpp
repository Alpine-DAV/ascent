//
// Created by Li, Jixian on 2019-06-04.
//

#ifndef ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H
#define ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H

#include <flow_filter.hpp>
#include <fstream>
#include <sstream>

namespace ascent
{
namespace runtime
{
namespace filters
{

// Xuan
// change here from op to op_enum to avoid naming dup
enum op_enum
{
  NOOP = 0,
  PMT
};

class BabelFlow : public ::flow::Filter
{
private:
  op_enum op = NOOP;

public:
  BabelFlow()= default;
  void declare_interface(conduit::Node &i) override;

  void execute() override;

  bool verify_params(const conduit::Node &params, conduit::Node &info) override;
};
}
}
}


#endif //ASCENT_ASCENT_RUNTIME_BABELFLOW_FILTERS_H
