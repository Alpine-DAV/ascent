#include "vtkmVectorMagnitude.hpp"

#include <vtkm/filter/vector_analysis/VectorMagnitude.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmVectorMagnitude::Run(vtkm::cont::DataSet &input,
                         std::string field_name,
                         std::string out_field_name,
                         vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::vector_analysis::VectorMagnitude mag;
  mag.SetActiveField(field_name);
  mag.SetOutputFieldName(out_field_name);
  mag.SetFieldsToPass(map_fields);

  auto output = mag.Execute(input);
  return output;
}

} // namespace vtkh
