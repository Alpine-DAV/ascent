#ifndef VTK_H_WARPX_STREAMLINE_HPP
#define VTK_H_WARPX_STREAMLINE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

#include <vtkm/Particle.h>

namespace vtkh
{

class VTKH_API WarpXStreamline : public Filter
{
public:
  WarpXStreamline();
  virtual ~WarpXStreamline();
  std::string GetName() const override { return "vtkh::WarpXStreamline";}
  void SetField(const std::string &field_name) {  m_field_name = field_name; }
  void SetStepSize(const double &step_size) {   m_step_size = step_size; }
  void SetNumberOfSteps(int numSteps) { m_num_steps = numSteps; }

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  double m_step_size;
  int m_num_steps;
};

} //namespace vtkh
#endif
