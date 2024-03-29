#ifndef VTK_H_SAMPLE_GRID_HPP
#define VTK_H_SAMPLE_GRID_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>


namespace vtkh
{

using Vec3f = vtkm::Vec<vtkm::Float64,3>;

class VTKH_API UniformGrid : public Filter
{
public:
  UniformGrid();
  virtual ~UniformGrid();
  std::string GetName() const override;
  void Dims(const Vec3f dims);
  void Origin(const Vec3f origin);
  void Spacing(const Vec3f spacing);
  void Field(const std::string field);
  void InvalidValue(const vtkm::Float64 invalid_value);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  Vec3f m_dims;
  Vec3f m_origin;
  Vec3f m_spacing;
  std::string m_field;
  vtkm::Float64 m_invalid_value;
};

} //namespace vtkh
#endif
