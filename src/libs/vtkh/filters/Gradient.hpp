#ifndef VTK_H_GRADIENT_HPP
#define VTK_H_GRADIENT_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

#include <vtkh/vtkm_filters/GradientParameters.hpp>

namespace vtkh
{

class VTKH_API Gradient : public Filter
{
public:
  Gradient();
  virtual ~Gradient();
  std::string GetName() const override;

  void SetField(const std::string &field_name);
  void SetParameters(GradientParameters params);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  GradientParameters m_params;
};

} //namespace vtkh
#endif
