#ifndef VTK_H_POINT_TRANSFORM_HPP
#define VTK_H_POINT_TRANSFORM_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/Matrix.h>

namespace vtkh
{

class VTKH_API PointTransform : public Filter
{
public:
  PointTransform();
  virtual ~PointTransform();
  std::string GetName() const override;

  void ResetTransform();

  void SetTranslation(const double& tx, const double& ty, const double& tz);
  void SetRotation(const double& angleDegrees, const vtkm::Vec<double, 3>& axis);
  void SetScale(const double& sx, const double& sy, const double& sz);
  void SetTransform(const vtkm::Matrix<double, 4, 4>& mtx);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::string m_field_name;
  vtkm::Matrix<vtkm::Float64, 4,4> m_transform;
};

} //namespace vtkh
#endif
