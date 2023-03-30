<<<<<<< HEAD
#ifndef VTK_H_VTKM_STATISTICS_HPP
#define VTK_H_VTKM_STATISTICS_HPP

#include <vtkm/cont/PartitionedDataSet.h>

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
