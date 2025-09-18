#include "viskoresClipWithField.hpp"
#include <viskores/filter/contour/ClipWithField.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresClipWithField::Run(viskores::cont::DataSet &input,
                       std::string field_name,
                       double clip_value,
                       bool invert,
                       viskores::filter::FieldSelection map_fields)
{
  viskores::filter::contour::ClipWithField clipper;

  clipper.SetClipValue(clip_value);
  clipper.SetInvertClip(invert);
  clipper.SetActiveField(field_name);
  clipper.SetFieldsToPass(map_fields);

  auto output = clipper.Execute(input);
  return output;
}

} // namespace vtkh
