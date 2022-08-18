#include "vtkmExtractStructured.hpp"
#include <vtkm/filter/entity_extraction/ExtractStructured.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmExtractStructured::Run(vtkm::cont::DataSet &input,
                           vtkm::RangeId3 range,
                           vtkm::Id3 sample_rate,
                           vtkm::filter::FieldSelection map_fields)
{

  vtkm::filter::entity_extraction::ExtractStructured extract;
  extract.SetVOI(range);
  extract.SetSampleRate(sample_rate);
  extract.SetIncludeBoundary(true);
  extract.SetFieldsToPass(map_fields);

  auto output = extract.Execute(input);
  return output;
}

} // namespace vtkh
