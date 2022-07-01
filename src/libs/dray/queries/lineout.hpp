#ifndef DRAY_LINEOUT_HPP
#define DRAY_LINEOUT_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class Lineout
{
protected:
  int32 m_samples;
  Float m_empty_val;
  std::vector<Vec<Float,3>> m_starts;
  std::vector<Vec<Float,3>> m_ends;
  std::vector<std::string> m_vars;
public:
  Lineout();

  struct Result
  {
    Array<Vec<Float,3>> m_points;
    int32 m_points_per_line;
    std::vector<Array<Float>> m_values;
    std::vector<std::string> m_vars;
    Float m_empty_val;
  };

  int32 samples() const;
  void samples(int32 samples);
  void empty_val(const Float val);
  void add_line(const Vec<Float,3> start, const Vec<Float,3> end);
  void add_var(const std::string var);
  Lineout::Result execute(Collection &collection);
  Array<Vec<Float,3>> create_points();
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP
