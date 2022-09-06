#ifndef VTK_H_VTKM_THRESHOLD_HPP
#define VTK_H_VTKM_THRESHOLD_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmThreshold
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          std::string field_name,
                          double min_value,
                          double max_value,
                          vtkm::filter::FieldSelection map_fields,
                          bool return_all_in_range = false);
};
}
#endif
