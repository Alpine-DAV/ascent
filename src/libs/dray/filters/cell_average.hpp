#ifndef DRAY_CELL_AVERAGE_HPP
#define DRAY_CELL_AVERAGE_HPP

#include <dray/data_model/collection.hpp>
#include <dray/error.hpp>

#include <string>

namespace dray
{

class CellAverage
{
protected:
  std::string in_field;
  std::string out_field;
public:
  CellAverage();
  ~CellAverage();

  void set_field(const std::string &name);
  void set_output_field(const std::string &name);

  Collection execute(Collection &);
};

}

#endif
