#ifndef VTK_H_VISKORES_LINEAR_EXTRUDE_HPP
#define VTK_H_VISKORES_LINEAR_EXTRUDE_HPP

#include <viskores/Types.h>
#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

// Extrude an input mesh along a vector to create a layered output dataset and optionally map fields.
class viskoresLinearExtrude
{
public:
  // Run the linear extrude operation for the requested number of steps.
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                              const viskores::Vec<viskores::Float64,3> &vector,
                              const viskores::Int32 steps,
                              viskores::filter::FieldSelection map_fields);
};

} // namespace vtkh

#endif
