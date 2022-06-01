#include "vtkmCellAverage.hpp"
#include <vtkm/filter/CellAverage.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmCellAverage::Run(vtkm::cont::DataSet &input,
                     std::string field_name,
                     std::string output_field_name,
                     vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::CellAverage avg;
  avg.SetOutputFieldName(output_field_name);
  avg.SetFieldsToPass(map_fields);
  avg.SetActiveField(field_name);

  auto output = avg.Execute(input);
  return output;
}

} // namespace vtkh
