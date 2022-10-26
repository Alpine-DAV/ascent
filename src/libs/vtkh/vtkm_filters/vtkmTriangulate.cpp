#include "vtkmTriangulate.hpp"

#include <vtkm/filter/geometry_refinement/Triangulate.h>

namespace vtkh
{

vtkm::cont::DataSet
vtkmTriangulate::Run(vtkm::cont::DataSet &input,
                     vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::geometry_refinement::Triangulate tri;
  tri.SetFieldsToPass(map_fields);
  auto output = tri.Execute(input);
  return output;
}

} // namespace vtkh
