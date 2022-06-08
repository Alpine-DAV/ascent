#include "vtkmClip.hpp"
#include <vtkm/filter/contour/ClipWithImplicitFunction.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmClip::Run(vtkm::cont::DataSet &input,
              const vtkm::ImplicitFunctionGeneral &func,
              bool invert,
              vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::contour::ClipWithImplicitFunction clipper;

  clipper.SetImplicitFunction(func);
  clipper.SetInvertClip(invert);
  clipper.SetFieldsToPass(map_fields);

  auto output = clipper.Execute(input);
  return output;
}

} // namespace vtkh
