#ifndef VTK_H_VTKM_HISTOGRAM_HPP
#define VTK_H_VTKM_HISTOGRAM_HPP

#include <vtkm/cont/PartitionedDataSet.h>

namespace vtkh
{

class vtkmHistogram
{
public:
  vtkm::cont::PartitionedDataSet Run(vtkm::cont::PartitionedDataSet &p_input,
                          vtkm::Id num_bins,
			  vtkm::Range range);
};
}
#endif
