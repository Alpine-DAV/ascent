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
  void Origin(const Vec3f origin);
  void Spacing(const Vec3f spacing);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  Vec3f m_origin;
  Vec3f m_spacing;
};

} //namespace vtkh
#endif
