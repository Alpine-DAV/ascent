#ifndef VTK_H_VISKORES_GHOSTSTRIPPER_HPP
#define VTK_H_VISKORES_GHOSTSTRIPPER_HPP

#include <viskores/cont/DataSet.h>

namespace vtkh
{

class viskoresGhostStripper
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          std::string ghost_field_name);
};
}
#endif
