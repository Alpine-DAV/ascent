#include "viskoresPointAverage.hpp"
#include <viskores/filter/field_conversion/PointAverage.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresPointAverage::Run(viskores::cont::DataSet &input,
                      std::string field_name,
                      std::string output_field_name,
                      viskores::filter::FieldSelection map_fields)
{
  viskores::filter::field_conversion::PointAverage avg;
  avg.SetOutputFieldName(output_field_name);
  avg.SetFieldsToPass(map_fields);
  avg.SetActiveField(field_name);

  auto output = avg.Execute(input);
  return output;
}

} // namespace vtkh
