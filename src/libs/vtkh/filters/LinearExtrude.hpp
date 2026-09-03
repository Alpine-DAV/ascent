#ifndef VTK_H_LINEAR_EXTRUDE_HPP
#define VTK_H_LINEAR_EXTRUDE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Filter.hpp>

#include <viskores/Types.h>

namespace vtkh
{

class VTKH_API LinearExtrude : public Filter
{
public:
  LinearExtrude();
  virtual ~LinearExtrude();
  std::string GetName() const override;

  void SetVector(const double vector[3]);
  void SetSteps(const int steps);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  viskores::Vec<viskores::Float64,3> m_vector;
  int m_steps;
};

} //namespace vtkh

#endif

