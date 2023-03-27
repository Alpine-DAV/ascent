#include "vtkmStatistics.hpp"
#include <vtkm/filter/density_estimate/Statistics.h>

namespace vtkh
{
vtkm::cont::PartitionedDataSet
vtkmStatistics::Run(vtkm::cont::PartitionedDataSet &p_input,
	      std::string field_name)
{
  vtkm::filter::density_estimate::Statistics stats;

  stats.SetActiveField(field_name);

  auto output = stats.Execute(p_input);
  return output;
}

} // namespace vtkh
