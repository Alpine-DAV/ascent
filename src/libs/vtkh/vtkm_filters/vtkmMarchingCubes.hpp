#ifndef VTK_H_VTKM_MARCHING_CUBES_HPP
#define VTK_H_VTKM_MARCHING_CUBES_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmMarchingCubes
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          std::string field_name,
                          std::vector<double> iso_values,
                          vtkm::filter::FieldSelection map_fields);
};
}
#endif
