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

  /**
   @brief Set the field name you wish to operate on
  */
  void set_field(const std::string &name);
  /**
   @brief Set the field name for the resulting cell-centered field.
          If empty (default), execute will use the same name for the output
          field as the input field.
  */
  void set_output_field(const std::string &name);

  /**
   @brief Compute the cell average value for the given field
   @return A collection with datasets containing the resulting
           cell-centered field.
  */
  Collection execute(Collection &);
};

}

#endif
