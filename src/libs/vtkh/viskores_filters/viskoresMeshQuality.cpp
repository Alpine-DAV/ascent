#include "viskoresMeshQuality.hpp"

namespace vtkh
{

viskores::cont::DataSet viskoresMeshQuality::Run(viskores::cont::DataSet &input,
                                         viskores::filter::mesh_info::CellMetric metric,
                                         viskores::filter::FieldSelection map_fields)

{
  viskores::filter::mesh_info::MeshQuality quali;
  quali.SetMetric(metric);
  quali.SetFieldsToPass(map_fields);
  auto output = quali.Execute(input);
  return output;
}

} // namespace vtkh
