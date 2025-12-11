#ifndef VTK_H_SAMPLE_GRID_HPP
#define VTK_H_SAMPLE_GRID_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>


namespace vtkh
{

using Vec3_f64    = viskores::Vec<viskores::Float64, 3>;

class VTKH_API UniformGrid : public Filter
{
public:
  UniformGrid();
  virtual ~UniformGrid();
  std::string GetName() const override;
  void Dims(const Vec3_f64 dims);
  void Origin(const Vec3_f64 origin);
  void Spacing(const Vec3_f64 spacing);
  void Fields(const std::vector<std::string> fields);
  void InvalidValue(const viskores::Float64 invalid_value);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  Vec3_f64 m_dims;
  Vec3_f64 m_origin;
  Vec3_f64 m_spacing;
  std::vector<std::string> m_fields;
  viskores::Float64 m_invalid_value;
};

} //namespace vtkh
#endif
