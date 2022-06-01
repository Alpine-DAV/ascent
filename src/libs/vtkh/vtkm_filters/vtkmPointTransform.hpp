#ifndef VTK_H_VTKM_POINT_TRANSFORM_HPP
#define VTK_H_VTKM_POINT_TRANSFORM_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/Matrix.h>
#include <vtkm/filter/FieldSelection.h>

namespace vtkh
{

class vtkmPointTransform
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          vtkm::Matrix<double,4,4> &transform,
                          vtkm::filter::FieldSelection map_fields);
};
}
#endif
