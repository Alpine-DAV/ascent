#ifndef VTK_H_VISKORES_THRESHOLD_HPP
#define VTK_H_VISKORES_THRESHOLD_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresThreshold
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          std::string field_name,
                          double min_value,
                          double max_value,
                          viskores::filter::FieldSelection map_fields,
                          bool return_all_in_range = false);
};
}
#endif
