#ifndef VTK_H_VTKM_PROBE_HPP
#define VTK_H_VTKM_PROBE_HPP

#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class vtkmProbe
{
protected:
  vtkm::Float64 m_x_origin;
  vtkm::Float64 m_y_origin;
  vtkm::Float64 m_z_origin;
  vtkm::Float64 m_x_spacing;
  vtkm::Float64 m_y_spacing;
  vtkm::Float64 m_z_spacing;
public:
  void x_origin(const vtkm::Float64 x);
  void y_origin(const vtkm::Float64 y);
  void z_origin(const vtkm::Float64 z);
  void x_spacing(const vtkm::Float64 dx);
  void y_spacing(const vtkm::Float64 dy);
  void z_spacing(const vtkm::Float64 dz);

  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input);
};
}
#endif
