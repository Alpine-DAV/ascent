#ifndef VTK_H_VTKM_SLICE_HPP
#define VTK_H_VTKM_SLICE_HPP

#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class vtkmSlice
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          vtkm::Vec<vtkm::Float32,3> m_points,
                          vtkm::Vec<vtkm::Float32,3> m_normals,
			  std::string fname);
};
}
#endif
