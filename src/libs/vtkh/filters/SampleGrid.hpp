#ifndef VTK_H_SAMPLE_GRID_HPP
#define VTK_H_SAMPLE_GRID_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>


namespace vtkh
{

using Vec3f = vtkm::Vec<vtkm::Float64,3>;

class VTKH_API SampleGrid : public Filter
{
public:
  SampleGrid();
  virtual ~SampleGrid();
  std::string GetName() const override;
  void Dims(const Vec3f dims);
  void Origin(const Vec3f origin);
  void Spacing(const Vec3f spacing);
  void InvalidValue(const vtkm::Float64 invalid_value);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  Vec3f m_dims;
  Vec3f m_origin;
  Vec3f m_spacing;
  vtkm::Float64 m_invalid_value;
};

} //namespace vtkh
#endif
