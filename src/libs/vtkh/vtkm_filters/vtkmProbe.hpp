#ifndef VTK_H_VTKM_PROBE_HPP
#define VTK_H_VTKM_PROBE_HPP

#include <vtkm/cont/DataSet.h>

namespace vtkh
{

using Vec3f = vtkm::Vec<vtkm::Float64,3>;

class vtkmProbe
{
protected:
  Vec3f m_dims;
  Vec3f m_origin;
  Vec3f m_spacing;
public:
  void dims(const Vec3f dims);
  void origin(const Vec3f origin);
  void spacing(const Vec3f spacing);

  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input);
};
}
#endif
