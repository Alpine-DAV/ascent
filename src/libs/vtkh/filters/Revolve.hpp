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
  ~Revolve() override;

  std::string GetName() const override;

  void SetAxis(const viskores::Vec3f &axis);
  void SetPoint(const viskores::Vec3f &point);
  void SetAngleDegrees(viskores::FloatDefault angle_degrees);
  void SetNumSteps(viskores::Id num_steps);
  void SetCapping(bool capping);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  viskores::Vec3f m_axis;
  viskores::Vec3f m_point;
  viskores::FloatDefault m_angle_degrees;
  viskores::Id m_num_steps;
  bool m_capping;
};

} // namespace vtkh

#endif
