#include "viskoresClip.hpp"
#include <viskores/filter/contour/ClipWithImplicitFunction.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresClip::Run(viskores::cont::DataSet &input,
              const viskores::ImplicitFunctionGeneral &func,
              bool invert,
              viskores::filter::FieldSelection map_fields)
{
  viskores::filter::contour::ClipWithImplicitFunction clipper;

  clipper.SetImplicitFunction(func);
  clipper.SetInvertClip(invert);
  clipper.SetFieldsToPass(map_fields);

  auto output = clipper.Execute(input);
  return output;
}

} // namespace vtkh
