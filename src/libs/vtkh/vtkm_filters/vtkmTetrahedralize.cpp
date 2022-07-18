#include "vtkmTetrahedralize.hpp"

#include <vtkm/filter/geometry_refinement/Tetrahedralize.h>

namespace vtkh
{

vtkm::cont::DataSet
vtkmTetrahedralize::Run(vtkm::cont::DataSet &input,
                        vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::geometry_refinement::Tetrahedralize tet;
  tet.SetFieldsToPass(map_fields);
  auto output = tet.Execute(input);
  return output;
}

} // namespace vtkh
