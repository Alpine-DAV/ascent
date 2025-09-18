#include "viskoresMarchingCubes.hpp"
#include <viskores/filter/contour/Contour.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresMarchingCubes::Run(viskores::cont::DataSet &input,
                       std::string field_name,
                       std::vector<double> iso_values,
                       viskores::filter::FieldSelection map_fields)
{
  viskores::filter::contour::Contour marcher;

  marcher.SetFieldsToPass(map_fields);
  marcher.SetIsoValues(iso_values);
  marcher.SetMergeDuplicatePoints(false);
  marcher.SetActiveField(field_name);

  auto output = marcher.Execute(input);
  return output;
}

} // namespace vtkh
