#ifndef VTK_H_VISKORES_REVOLVE_HPP
#define VTK_H_VISKORES_REVOLVE_HPP

#include <viskores/Types.h>
#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresRevolve
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                              viskores::filter::FieldSelection map_fields,
                              const viskores::Vec3f &axis,
                              const viskores::Vec3f &point,
                              viskores::FloatDefault angle_degrees,
                              viskores::Id num_steps,
                              bool capping);
};

} // namespace vtkh

#endif
