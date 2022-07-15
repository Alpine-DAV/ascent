#ifndef VTK_H_VTKM_EXTRACT_STRUCTURED_HPP
#define VTK_H_VTKM_EXTRACT_STRUCTURED_HPP

#include <vtkm/RangeId3.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmExtractStructured
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                           vtkm::RangeId3 range,
                           vtkm::Id3 sample_rate,
                           vtkm::filter::FieldSelection map_fields);
};
}
#endif
