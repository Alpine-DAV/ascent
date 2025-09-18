#include "viskoresTriangulate.hpp"

#include <viskores/filter/geometry_refinement/Triangulate.h>

namespace vtkh
{

viskores::cont::DataSet
viskoresTriangulate::Run(viskores::cont::DataSet &input,
                     viskores::filter::FieldSelection map_fields)
{
  viskores::filter::geometry_refinement::Triangulate tri;
  tri.SetFieldsToPass(map_fields);
  auto output = tri.Execute(input);
  return output;
}

} // namespace vtkh
