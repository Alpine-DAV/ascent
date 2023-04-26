#ifndef VTK_H_VTKM_GHOSTSTRIPPER_HPP
#define VTK_H_VTKM_GHOSTSTRIPPER_HPP

#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class vtkmGhostStripper
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          std::string ghost_field_name);
};
}
#endif
