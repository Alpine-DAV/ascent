#include "vtkmProbe.hpp"
#include <vtkm/filter/resampling/Probe.h>

namespace vtkh
{

void
vtkmProbe::x_origin(const vtkm::Float64 x)
{
  m_x_origin = x;
}
void
vtkmProbe::y_origin(const vtkm::Float64 y)
{
  m_y_origin = y;
}
void
vtkmProbe::z_origin(const vtkm::Float64 z)
{
  m_z_origin = z;
}
void
vtkmProbe::x_spacing(const vtkm::Float64 x)
{
  m_x_spacing = x;
}
void
vtkmProbe::y_spacing(const vtkm::Float64 y)
{
  m_y_spacing = y;
}
void
vtkmProbe::z_spacing(const vtkm::Float64 z)
{
  m_z_spacing = z;
}

vtkm::cont::DataSet
vtkmProbe::Run(vtkm::cont::DataSet &input)
{
  vtkm::filter::resampling::Probe probe;
  vtkm::cont::DataSet ds_probe;

  std::string name = "coords";
  int dims = 3;
  if(m_z_spacing == 0)
    dims = 2;

  vtkm::Vec3f origin(m_x_origin,m_y_origin,m_z_origin);
  vtkm::Vec3f spacing(m_x_spacing,m_y_spacing,m_z_spacing);

  vtkm::cont::CoordinateSystem cs(name, dims, origin, spacing);
  ds_probe.AddCoordinateSystem(cs);
  probe.SetGeometry(ds_probe);

  auto output = probe.Execute(input);
  return output;
}

} // namespace vtkh
