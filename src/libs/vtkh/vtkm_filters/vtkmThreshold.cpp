#include "viskoresThreshold.hpp"

#include <viskores/filter/entity_extraction/Threshold.h>
#include <viskores/cont/CellSetPermutation.h>


namespace vtkh
{

viskores::cont::DataSet
viskoresThreshold::Run(viskores::cont::DataSet &input,
                   std::string field_name,
                   double min_value,
                   double max_value,
                   viskores::filter::FieldSelection map_fields,
                   bool return_all_in_range)
{
  viskores::filter::entity_extraction::Threshold thresholder;
  thresholder.SetAllInRange(return_all_in_range);
  thresholder.SetUpperThreshold(max_value);
  thresholder.SetLowerThreshold(min_value);
  thresholder.SetActiveField(field_name);
  thresholder.SetFieldsToPass(map_fields);
  auto output = thresholder.Execute(input);
  
  return output;
}

} // namespace vtkh
