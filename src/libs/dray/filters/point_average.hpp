#ifndef DRAY_POINT_AVERAGE_HPP
#define DRAY_POINT_AVERAGE_HPP

#include <dray/data_model/collection.hpp>
#include <dray/error.hpp>

#include <string>

namespace dray
{

class PointAverage
{
protected:
  std::string in_field;
  std::string out_field;
  std::string ref_mesh;
public:
  PointAverage();
  ~PointAverage();

  /**
   @brief Set the field name you wish to operate on
  */
  void set_field(const std::string &name);

  /**
   @brief Set the field name for the resulting point-centered field.
          If empty (default), execute will use the same name for the output
          field as the input field.
  */
  void set_output_field(const std::string &name);

  /**
   @brief Set the name of the mesh to use.
          If empty (default), execute will use the first mesh in each dataset.
  */
  void set_mesh(const std::string &name);

  /**
   @brief Compute the point average value for the given field
   @return A dataset containing the resulting point-centered field
  */
  DataSet execute(DataSet &);

  /**
   @brief Compute the point average value for the given field
   @return A collection with datasets containing the resulting
           point-centered field.
  */
  Collection execute(Collection &);
};

}

#endif
