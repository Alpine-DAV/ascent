#ifndef VTK_H_VISKORES_CLIP_WITH_FIELD_HPP
#define VTK_H_VISKORES_CLIP_WITH_FIELD_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresClipWithField
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                       std::string field_name,
                       double clip_value,
                       bool invert,
                       viskores::filter::FieldSelection map_fields);
};
}
#endif
