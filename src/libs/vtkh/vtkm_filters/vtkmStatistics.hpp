#ifndef VTK_H_VTKM_STATISTICS_HPP
#define VTK_H_VTKM_STATISTICS_HPP

#include <vtkm/cont/PartitionedDataSet.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/density_estimate/Statistics.h>

namespace vtkh
{

class vtkmStatistics
{
public:
  vtkm::cont::PartitionedDataSet Run(vtkm::cont::PartitionedDataSet &p_input,
		                     std::string field_name);
};
}
#endif
