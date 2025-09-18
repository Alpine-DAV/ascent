#include "viskoresStatistics.hpp"
#include <viskores/filter/density_estimate/Statistics.h>

namespace vtkh
{
viskores::cont::PartitionedDataSet
viskoresStatistics::Run(viskores::cont::PartitionedDataSet &p_input,
	      std::string field_name)
{
  viskores::filter::density_estimate::Statistics stats;

  stats.SetActiveField(field_name);

  auto output = stats.Execute(p_input);
  //output.PrintSummary(std::cerr);
  return output;
}

} // namespace vtkh
