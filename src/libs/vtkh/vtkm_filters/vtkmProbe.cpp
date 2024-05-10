#include "vtkmProbe.hpp"
#include <vtkm/filter/resampling/Probe.h>
#include <vtkm/cont/DataSetBuilderUniform.h>

namespace vtkh
{

void
vtkmProbe::dims(const Vec3f dims)
{
  m_dims = dims;
}

void
vtkmProbe::origin(const Vec3f origin)
{
  m_origin = origin;
}

void
vtkmProbe::spacing(const Vec3f spacing)
{
  m_spacing = spacing;
}

void
vtkmProbe::invalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

vtkm::cont::DataSet
vtkmProbe::Run(vtkm::cont::DataSet &input)
{
  vtkm::filter::resampling::Probe probe;
  vtkm::cont::DataSet ds_probe;

  if(m_dims[2] <= 1)
  {
    Vec2f t_dims = {m_dims[0],m_dims[1]};
    Vec2f t_origin = {m_origin[0],m_origin[1]};
    Vec2f t_spacing = {m_spacing[0],m_spacing[1]};
    ds_probe = vtkm::cont::DataSetBuilderUniform::Create(t_dims, t_origin, t_spacing);
  }
  else
  {
    ds_probe = vtkm::cont::DataSetBuilderUniform::Create(m_dims, m_origin, m_spacing);
  }
  probe.SetGeometry(ds_probe);
  probe.SetInvalidValue(m_invalid_value);

  auto output = probe.Execute(input);
//  std::cerr << "RESULT START" << std::endl;
//  output.PrintSummary(std::cerr);
//  std::cerr << "RESULT END" << std::endl;
  return output;
}

} // namespace vtkh
