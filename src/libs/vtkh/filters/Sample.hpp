#ifndef VTK_H_SAMPLE_HPP
#define VTK_H_SAMPLE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>


namespace vtkh
{

using Scalar_i32_hnd = viskores::cont::ArrayHandle<viskores::Int32>;
using Scalar_f32_hnd = viskores::cont::ArrayHandle<viskores::Float32>;
using Scalar_f64_hnd = viskores::cont::ArrayHandle<viskores::Float64>;

using Vec2_f32_hnd  = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32,2>>;
using Vec2_f64_hnd  = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64,2>>;

using Vec3_f32_hnd  = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float32,3>>;
using Vec3_f64_hnd  = viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64,3>>;

using Vec2_f32    = viskores::Vec<viskores::Float32, 2>;
using Vec3_f32    = viskores::Vec<viskores::Float32, 3>;

using Vec2_f64    = viskores::Vec<viskores::Float64, 2>;
using Vec3_f64    = viskores::Vec<viskores::Float64, 3>;


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

  void Points(viskores::cont::ArrayHandle<viskores::Float64> xs,
              viskores::cont::ArrayHandle<viskores::Float64> ys,
              viskores::cont::ArrayHandle<viskores::Float64> zs);

  void UniformGrid(const Vec3_f64 dims,
                   const Vec3_f64 origin,
                   const Vec3_f64 spacing);

  void Fields(const std::vector<std::string> fields);
  void InvalidValue(const viskores::Float64 invalid_value);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  viskores::cont::ArrayHandle<viskores::Float64> m_points_xs;
  viskores::cont::ArrayHandle<viskores::Float64> m_points_ys;
  viskores::cont::ArrayHandle<viskores::Float64> m_points_zs;

  std::vector<std::string> m_fields;
  viskores::Float64 m_invalid_value;
  Vec3_f64 m_dims;
  Vec3_f64 m_origin;
  Vec3_f64 m_spacing;
  int m_num_samples;
  bool m_is_points;
};

} //namespace vtkh
#endif
