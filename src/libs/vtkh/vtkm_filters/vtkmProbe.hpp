#ifndef VTK_H_VTKM_PROBE_HPP
#define VTK_H_VTKM_PROBE_HPP

#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class vtkmProbe
{
public:
  vtkmProbe();
  ~vtkmProbe();

  void setPoints(vtkm::cont::ArrayHandle<vtkm::Float64> xs,
                 vtkm::cont::ArrayHandle<vtkm::Float64> ys,
                 vtkm::cont::ArrayHandle<vtkm::Float64> zs);

  void setBoxDims(const vtkm::Vec<vtkm::Float64,3> dims);
  void setBoxOrigin(const vtkm::Vec<vtkm::Float64,3> origin);
  void setBoxSpacing(const vtkm::Vec<vtkm::Float64,3> spacing);

  void setInvalidValue(const vtkm::Float64 invalid_value);

  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input);

protected:
  enum SampleMode { 
                    NONE,
                    POINTS,
                    BOX
                  };

  int m_mode;

  vtkm::Float64 m_invalid_value;

  // points details
  vtkm::cont::ArrayHandle<vtkm::Float64> m_points_xs;
  vtkm::cont::ArrayHandle<vtkm::Float64> m_points_ys;
  vtkm::cont::ArrayHandle<vtkm::Float64> m_points_zs;

  // box details
  vtkm::Vec<vtkm::Float64,3> m_box_dims;
  vtkm::Vec<vtkm::Float64,3> m_box_origin;
  vtkm::Vec<vtkm::Float64,3> m_box_spacing;


};
}
#endif
