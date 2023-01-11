#ifndef VTK_H_VTKM_CLIP_HPP
#define VTK_H_VTKM_CLIP_HPP

#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class vtkmHistogram
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::Field &input,
                          vtkm::Id num_bins,
			  vtkm::Range range);
};
}
#endif
