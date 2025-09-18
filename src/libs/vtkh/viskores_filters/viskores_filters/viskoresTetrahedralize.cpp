#include "viskoresTetrahedralize.hpp"

#include <viskores/filter/geometry_refinement/Tetrahedralize.h>

namespace vtkh
{

viskores::cont::DataSet
viskoresTetrahedralize::Run(viskores::cont::DataSet &input,
                        viskores::filter::FieldSelection map_fields)
{
  viskores::filter::geometry_refinement::Tetrahedralize tet;
  tet.SetFieldsToPass(map_fields);
  auto output = tet.Execute(input);
  return output;
}

} // namespace vtkh
