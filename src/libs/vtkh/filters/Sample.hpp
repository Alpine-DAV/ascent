#ifndef VTK_H_SAMPLE_HPP
#define VTK_H_SAMPLE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>


namespace vtkh
{



class VTKH_API Sample : public Filter
{
public:
  Sample();
  virtual ~Sample();
  std::string GetName() const override;

  void Line(int num_samples,
            double x_start,
            double y_start,
            double z_start,
            double x_end,
            double y_end,
            double z_end);

  void Box(int * dims,
           double x_start,
           double y_start,
           double z_start,
           double x_end,
           double y_end,
           double z_end);

  void Points(vtkm::cont::ArrayHandle<vtkm::Float64> xs,
              vtkm::cont::ArrayHandle<vtkm::Float64> ys,
              vtkm::cont::ArrayHandle<vtkm::Float64> zs);

  // TODO: expand to uniform grid case (box)

  void Fields(const std::vector<std::string> fields);
  void InvalidValue(const vtkm::Float64 invalid_value);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  vtkm::cont::ArrayHandle<vtkm::Float64> m_points_xs;
  vtkm::cont::ArrayHandle<vtkm::Float64> m_points_ys;
  vtkm::cont::ArrayHandle<vtkm::Float64> m_points_zs;

  std::vector<std::string> m_fields;
  vtkm::Float64 m_invalid_value;
};

} //namespace vtkh
#endif
