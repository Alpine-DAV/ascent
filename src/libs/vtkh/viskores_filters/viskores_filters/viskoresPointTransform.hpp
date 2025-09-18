#ifndef VTK_H_VISKORES_POINT_TRANSFORM_HPP
#define VTK_H_VISKORES_POINT_TRANSFORM_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/Matrix.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresPointTransform
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          viskores::Matrix<double,4,4> &transform,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
