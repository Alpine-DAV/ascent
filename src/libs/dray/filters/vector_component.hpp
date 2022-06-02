#ifndef DRAY_VECTOR_COMPONENT_HPP
#define DRAY_VECTOR_COMPONENT_HPP

#include <dray/data_model/collection.hpp>

namespace dray
{

class VectorComponent
{
protected:
  int32 m_component;
  std::string m_field_name;
  std::string m_output_name;
public:
  VectorComponent();
  void component(const int32 comp);
  void output_name(const std::string &name);
  void field(const std::string &name);
  Collection execute(Collection &collection);
  // utility methods
  static std::shared_ptr<Field> execute(Field *field, const int32 comp);
  // break up all vector fields into component of the form
  // name_x, name_y ...
  static DataSet decompose_all(DataSet &input);
  static DataSet decompose_field(DataSet &input, const std::string &field_name);
  static Collection decompose_all(Collection &input);
  static Collection decompose_field(Collection &input, const std::string &field_name);
};

};//namespace dray

#endif//DRAY_MESH_BOUNDARY_HPP
