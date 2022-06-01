#ifndef VTK_VECTOR_COMPONENT_HPP
#define VTK_VECTOR_COMPONENT_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API VectorComponent : public Filter
{
public:
  VectorComponent();
  virtual ~VectorComponent();
  std::string GetName() const override;
  void SetField(const std::string &field);
  void SetComponent(const int component);

  void SetResultField(const std::string &result_name);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  int m_component;
  std::string m_field_name;
  std::string m_result_name;
};

} //namespace vtkh
#endif
