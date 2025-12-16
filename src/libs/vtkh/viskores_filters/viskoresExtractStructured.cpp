#include "viskoresExtractStructured.hpp"
#include <viskores/filter/entity_extraction/ExtractStructured.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresExtractStructured::Run(viskores::cont::DataSet &input,
                           viskores::RangeId3 range,
                           viskores::Id3 sample_rate,
                           viskores::filter::FieldSelection map_fields)
{

  viskores::filter::entity_extraction::ExtractStructured extract;
  extract.SetVOI(range);
  extract.SetSampleRate(sample_rate);
  extract.SetIncludeBoundary(true);
  extract.SetFieldsToPass(map_fields);

  auto output = extract.Execute(input);
  return output;
}

} // namespace vtkh
