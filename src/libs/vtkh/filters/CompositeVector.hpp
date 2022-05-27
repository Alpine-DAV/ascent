#ifndef VTK_COMPOSITE_VECTOR_HPP
#define VTK_COMPOSITE_VECTOR_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API CompositeVector : public Filter
{
public:
  CompositeVector();
  virtual ~CompositeVector();
  std::string GetName() const override;
  void SetFields(const std::string &field1,
                 const std::string &field2,
                 const std::string &field3);

  void SetFields(const std::string &field1,
                 const std::string &field2);

  void SetResultField(const std::string &result_name);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_1;
  std::string m_field_2;
  std::string m_field_3;
  std::string m_result_name;
  bool m_mode_3d;
};

} //namespace vtkh
#endif
