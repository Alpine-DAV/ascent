#include "viskoresVectorMagnitude.hpp"

#include <viskores/filter/vector_analysis/VectorMagnitude.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresVectorMagnitude::Run(viskores::cont::DataSet &input,
                         std::string field_name,
                         std::string out_field_name,
                         viskores::filter::FieldSelection map_fields)
{
  viskores::filter::vector_analysis::VectorMagnitude mag;
  mag.SetActiveField(field_name);
  mag.SetOutputFieldName(out_field_name);
  mag.SetFieldsToPass(map_fields);

  auto output = mag.Execute(input);
  return output;
}

} // namespace vtkh
