#include "viskoresPointTransform.hpp"
#include <viskores/filter/field_transform/PointTransform.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresPointTransform::Run(viskores::cont::DataSet &input,
                        viskores::Matrix<double,4,4> &transform,
                        viskores::filter::FieldSelection map_fields)
{
  auto i_bounds = input.GetCoordinateSystem(0).GetBounds();
  viskores::filter::field_transform::PointTransform trans;

  trans.SetChangeCoordinateSystem(true);
  trans.SetFieldsToPass(map_fields);
  trans.SetTransform(transform);

  auto output = trans.Execute(input);
  auto bounds = output.GetCoordinateSystem(0).GetBounds();
  return output;
}

} // namespace vtkh
