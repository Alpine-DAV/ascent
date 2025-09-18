#ifndef VTK_H_VISKORES_MARCHING_CUBES_HPP
#define VTK_H_VISKORES_MARCHING_CUBES_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>

namespace vtkh
{

class viskoresMarchingCubes
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          std::string field_name,
                          std::vector<double> iso_values,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
