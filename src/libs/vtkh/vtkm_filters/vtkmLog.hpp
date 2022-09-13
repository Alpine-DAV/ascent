#ifndef VTK_H_VTKM_LOG_HPP
#define VTK_H_VTKM_LOG_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>
#include <vtkm/filter/field_transform/LogValues.h>

namespace vtkh
{

class vtkmLog
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          const LogBase base,
                          const vtkm::Float32 min_value);
};
}
#endif

