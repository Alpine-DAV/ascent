#ifndef VTK_H_VTKM_VECTOR_MAGNITUDE_HPP
#define VTK_H_VTKM_VECTOR_MAGNITUDE_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmVectorMagnitude
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          std::string field_name,
                          std::string out_field_name,
                          vtkm::filter::FieldSelection map_fields);
};
}
#endif
