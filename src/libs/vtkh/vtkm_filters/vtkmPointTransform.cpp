#include "vtkmPointTransform.hpp"
#include <vtkm/filter/field_transform/PointTransform.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmPointTransform::Run(vtkm::cont::DataSet &input,
                        vtkm::Matrix<double,4,4> &transform,
                        vtkm::filter::FieldSelection map_fields)
{
  vtkm::filter::field_transform::PointTransform trans;

  trans.SetChangeCoordinateSystem(true);
  trans.SetFieldsToPass(map_fields);
  trans.SetTransform(transform);

  auto output = trans.Execute(input);
  return output;
}

} // namespace vtkh
