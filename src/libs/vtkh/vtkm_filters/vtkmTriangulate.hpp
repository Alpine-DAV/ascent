#ifndef VTK_H_VTKM_TRIANGULATE_HPP
#define VTK_H_VTKM_TRIANGULATE_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmTriangulate
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          vtkm::filter::FieldSelection map_fields);
};
}
#endif
