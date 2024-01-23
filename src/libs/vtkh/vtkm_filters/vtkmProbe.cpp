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

  std::string name = "coords";
  vtkm::Id3 dims = 3;
  if(m_dims[2] == 0)
    dims = 2;

  vtkm::cont::DataSet ds_probe = vtkm::cont::DataSetBuilderUniform::Create(m_dims, m_origin, m_spacing);
  probe.SetGeometry(ds_probe);
  probe.SetInvalidValue(m_invalid_value);

  auto output = probe.Execute(input);
  return output;
}

} // namespace vtkh
