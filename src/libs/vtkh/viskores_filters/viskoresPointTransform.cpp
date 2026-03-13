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
  std::cerr << "old bounds: " << i_bounds.X.Min << " " << i_bounds.X.Max << " " << i_bounds.Y.Min << " " << i_bounds.Y.Max << " " << i_bounds.Z.Min << " " << i_bounds.Z.Max << std::endl;
  viskores::filter::field_transform::PointTransform trans;
  std::cerr << "transform: " << transform << std::endl;

  trans.SetChangeCoordinateSystem(true);
  trans.SetFieldsToPass(map_fields);
  trans.SetTransform(transform);

  auto output = trans.Execute(input);
  auto bounds = output.GetCoordinateSystem(0).GetBounds();
  std::cerr << "new bounds: " << bounds.X.Min << " " << bounds.X.Max << " " << bounds.Y.Min << " " << bounds.Y.Max << " " << bounds.Z.Min << " " << bounds.Z.Max << std::endl;
  return output;
}

} // namespace vtkh
