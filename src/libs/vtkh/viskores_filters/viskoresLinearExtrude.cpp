#include "viskoresLinearExtrude.hpp"

#include <viskores/Math.h>
#include <viskores/cont/ErrorFilterExecution.h>
#include <viskores/filter/geometry_refinement/ExtrusionLinear.h>

namespace vtkh
{

// Linearly extrude an input mesh along a vector for a fixed number of steps.
viskores::cont::DataSet
viskoresLinearExtrude::Run(viskores::cont::DataSet &input,
                           const viskores::Vec<viskores::Float64,3> &vector,
                           const viskores::Int32 steps,
                           const bool triangulate_input,
                           viskores::filter::FieldSelection map_fields)
{
  if (steps <= 0)
  {
    throw viskores::cont::ErrorFilterExecution("vtkh::LinearExtrude requires 'steps' > 0");
  }

  const viskores::Float64 perStepDistance = viskores::Magnitude(vector) / static_cast<viskores::Float64>(steps);

  viskores::filter::geometry_refinement::ExtrusionLinear extruder;
  extruder.SetFieldsToPass(map_fields);
  extruder.SetTriangulateInput(triangulate_input);
  extruder.SetCompactOutput(false);
  extruder.SetNumberOfPlanes(static_cast<viskores::Id>(steps) + 1);
  extruder.SetDirection(viskores::Vec3f(static_cast<viskores::FloatDefault>(vector[0]),
                                        static_cast<viskores::FloatDefault>(vector[1]),
                                        static_cast<viskores::FloatDefault>(vector[2])));
  extruder.SetDistance(static_cast<viskores::FloatDefault>(perStepDistance));

  return extruder.Execute(input);
}

} // namespace vtkh
