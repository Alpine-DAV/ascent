#include "vtkmThreshold.hpp"

#include <vtkm/filter/entity_extraction/Threshold.h>
#include <vtkm/cont/CellSetPermutation.h>


namespace vtkh
{
typedef vtkm::cont::CellSetPermutation<vtkm::cont::CellSetStructured<2>>
  PermStructured2d;

typedef vtkm::cont::CellSetPermutation<vtkm::cont::CellSetStructured<3>>
  PermStructured3d;

typedef vtkm::cont::CellSetPermutation<vtkm::cont::CellSetExplicit<>>
  PermExplicit;

typedef  vtkm::cont::CellSetPermutation<vtkm::cont::CellSetSingleType<>>
  PermExplicitSingle;

vtkm::cont::DataSet
vtkmThreshold::Run(vtkm::cont::DataSet &input,
                   std::string field_name,
                   double min_value,
                   double max_value,
                   vtkm::filter::FieldSelection map_fields,
                   bool return_all_in_range)
{
  vtkm::filter::entity_extraction::Threshold thresholder;
  thresholder.SetAllInRange(return_all_in_range);
  thresholder.SetUpperThreshold(max_value);
  thresholder.SetLowerThreshold(min_value);
  thresholder.SetActiveField(field_name);
  thresholder.SetFieldsToPass(map_fields);
  auto output = thresholder.Execute(input);
  
  return output;
}

} // namespace vtkh
