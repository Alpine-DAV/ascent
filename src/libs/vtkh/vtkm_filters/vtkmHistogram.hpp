#ifndef VTK_H_VISKORES_HISTOGRAM_HPP
#define VTK_H_VISKORES_HISTOGRAM_HPP

#include <viskores/cont/PartitionedDataSet.h>

namespace vtkh
{

class viskoresHistogram
{
public:
  viskores::cont::PartitionedDataSet Run(viskores::cont::PartitionedDataSet &p_input,
                          viskores::Id num_bins,
			  viskores::Range range);
};
}
#endif
