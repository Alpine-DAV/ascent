#ifndef VTK_H_VTKM_POINT_AVERAGE_HPP
#define VTK_H_VTKM_POINT_AVERAGE_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmPointAverage
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                     std::string field_name,
                     std::string output_field_name,
                     vtkm::filter::FieldSelection map_fields);
};
}
#endif
