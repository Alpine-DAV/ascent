#include "vtkmProbe.hpp"
#include <vtkm/filter/resampling/Probe.h>

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

vtkm::cont::DataSet
vtkmProbe::Run(vtkm::cont::DataSet &input)
{
  vtkm::filter::resampling::Probe probe;
  vtkm::cont::DataSet ds_probe;

  std::string name = "coords";
  int dims = 3;
  if(m_dims[2] == 0)
    dims = 2;

  vtkm::cont::CoordinateSystem cs(name, dims, m_origin, m_spacing);
  ds_probe.AddCoordinateSystem(cs);
  probe.SetGeometry(ds_probe);
  std::cerr << "INPUT VTKM DATA " << std::endl;
  input.PrintSummary(std::cerr);
  std::cerr << "END INPUT VTKM DATA " << std::endl;
  std::cerr << std::endl;
  std::cerr << "INPUT GEOMETRY" << std::endl;
  ds_probe.PrintSummary(std::cerr);
  std::cerr << "END INPUT GEOMETRY" << std::endl;

  std::cerr << "BEFORE VTKM EXECUTE" << std::endl;
  auto output = probe.Execute(input);
  std::cerr << "AFTER VTKM EXECUTE" << std::endl;
  return output;
}

} // namespace vtkh
