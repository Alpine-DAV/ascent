#ifndef VTK_H_VISKORES_CLIP_HPP
#define VTK_H_VISKORES_CLIP_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>
#include <viskores/ImplicitFunction.h>

namespace vtkh
{

class viskoresClip
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          const viskores::ImplicitFunctionGeneral &func,
                          bool invert,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
