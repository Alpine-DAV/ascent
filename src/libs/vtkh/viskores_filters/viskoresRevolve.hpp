#ifndef VTK_H_VISKORES_REVOLVE_HPP
#define VTK_H_VISKORES_REVOLVE_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>
#include <viskores/Types.h>

namespace vtkh
{

// Revolve an input mesh around an axis, producing a swept dataset and optionally mapping fields.
class viskoresRevolve
{
public:
  // Run the revolve operation, sweeping by the given angle and number of steps.
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                              const viskores::Vec<viskores::Float64,3> &point,
                              const viskores::Vec<viskores::Float64,3> &axis,
                              const viskores::Float64 start_angle_degrees,
                              const viskores::Float64 sweep_angle_degrees,
                              const viskores::Int32 steps,
                              const bool periodic,
                              viskores::filter::FieldSelection map_fields);
};

}

#endif
