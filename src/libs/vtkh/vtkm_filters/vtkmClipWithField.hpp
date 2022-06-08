#ifndef VTK_H_VTKM_CLIP_WITH_FIELD_HPP
#define VTK_H_VTKM_CLIP_WITH_FIELD_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmClipWithField
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                       std::string field_name,
                       double clip_value,
                       bool invert,
                       vtkm::filter::FieldSelection map_fields);
};
}
#endif
