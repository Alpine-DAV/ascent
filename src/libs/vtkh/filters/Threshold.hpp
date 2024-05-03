#ifndef VTK_H_THRESHOLD_HPP
#define VTK_H_THRESHOLD_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/Range.h>

#include <memory>

namespace vtkh
{

class VTKH_API Threshold: public Filter
{
public:
  Threshold();
  virtual ~Threshold();
  std::string GetName() const override;

  void SetAllInRange(const bool &value);
  bool GetAllInRange() const;
  std::string GetThresholdMode() const;

  // threshold by field
  void SetFieldUpperThreshold(const double &value);
  void SetFieldLowerThreshold(const double &value);
  void SetField(const std::string &field_name);

  // threshold by implicit function

  // invert
  void SetInvertThreshold(bool invert);

  // boundary
  void SetBoundaryThreshold(bool boundary);

  // threshold by box
  void SetBoxThreshold(const vtkm::Bounds &box_bounds);

  // threshold by plane
  void SetPlaneThreshold(const double plane_origin[3],
                         const double plane_normal[3]);

  // threshold by cylinder
  void SetCylinderThreshold(const double cylinder_center[3],
                            const double cylinder_axis[3],
                            const double cylinder_radius);

  // threshold by Sphere
  void SetSphereThreshold(const double sphere_center[3],
                          const double sphere_radius);



protected:
  void PreExecute()  override;
  void PostExecute() override;
  void DoExecute()   override;

  // for vtkm implicit fun for non field cases
  struct Internals;
  std::shared_ptr<Internals> m_internals;

};

} //namespace vtkh
#endif
