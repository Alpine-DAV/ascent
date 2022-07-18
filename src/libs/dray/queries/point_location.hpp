#ifndef DRAY_POINT_LOCATION_HPP
#define DRAY_POINT_LOCATION_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class PointLocation
{
protected:
  Float m_empty_val;
  std::vector<std::string> m_vars;
public:
  PointLocation();

  struct Result
  {
    Array<Vec<Float,3>> m_points;
    std::vector<Array<Float>> m_values;
    std::vector<std::string> m_vars;
    Float m_empty_val;
  };

  void empty_val(const Float val);
  void add_var(const std::string var);
  PointLocation::Result execute(Collection &collection, Array<Vec<Float,3>> &points);
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP
