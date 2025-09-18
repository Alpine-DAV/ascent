#ifndef VTK_H_VISKORES_STATISTICS_HPP
#define VTK_H_VISKORES_STATISTICS_HPP

#include <viskores/cont/PartitionedDataSet.h>
#include <viskores/cont/DataSet.h>
#include <viskores/filter/density_estimate/Statistics.h>

namespace vtkh
{

class viskoresStatistics
{
public:
  viskores::cont::PartitionedDataSet Run(viskores::cont::PartitionedDataSet &p_input,
		                     std::string field_name);
};
}
#endif
