#include "vtkmMeshQuality.hpp"

namespace vtkh
{

vtkm::cont::DataSet vtkmMeshQuality::Run(vtkm::cont::DataSet &input,
                                         vtkm::filter::mesh_info::CellMetric metric,
                                         vtkm::filter::FieldSelection map_fields)

{
  vtkm::filter::mesh_info::MeshQuality quali(metric);
  quali.SetFieldsToPass(map_fields);
  auto output = quali.Execute(input);
  return output;
}

} // namespace vtkh
