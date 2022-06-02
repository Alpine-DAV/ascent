#include "vtkmClipWithField.hpp"
#include <vtkm/filter/ClipWithField.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmClipWithField::Run(vtkm::cont::DataSet &input,
                       std::string field_name,
                       double clip_value,
                       bool invert,
                       vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::ClipWithField clipper;

  clipper.SetClipValue(clip_value);
  clipper.SetInvertClip(invert);
  clipper.SetActiveField(field_name);
  clipper.SetFieldsToPass(map_fields);

  auto output = clipper.Execute(input);
  return output;
}

} // namespace vtkh
