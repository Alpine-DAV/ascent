#ifndef VTK_H_VECTOR_MAGNITUDE_HPP
#define VTK_H_VECTOR_MAGNITUDE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API VectorMagnitude : public Filter
{
public:
  VectorMagnitude();
  virtual ~VectorMagnitude();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetResultName(const std::string name);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  std::string m_out_name;
};

} //namespace vtkh
#endif
