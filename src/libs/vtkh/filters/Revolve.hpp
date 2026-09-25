#ifndef VTK_H_REVOLVE_HPP
#define VTK_H_REVOLVE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Filter.hpp>
#include <viskores/Types.h>

namespace vtkh
{

class VTKH_API Revolve : public Filter
{
public:
  Revolve();
  virtual ~Revolve();
  std::string GetName() const override;

  void SetPoint(const double point[3]);
  void SetAxis(const double axis[3]);
  void SetStartAngle(const double start_angle_degrees);
  void SetSweepAngle(const double sweep_angle_degrees);
  void SetSteps(const int steps);
  void SetPeriodic(const bool periodic);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  viskores::Vec<viskores::Float64,3> m_point;
  viskores::Vec<viskores::Float64,3> m_axis;
  double m_start_angle_degrees;
  double m_sweep_angle_degrees;
  int m_steps;
  bool m_periodic;
};

} //namespace vtkh
#endif
