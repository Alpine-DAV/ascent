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

  std::cerr << "vtkm STATS Pre Execute" << std::endl;
  std::cerr << "data going in pre Execute: " << std::endl;
  p_input.PrintSummary(std::cerr);
  auto output = stats.Execute(p_input);
  std::cerr << "vtkm STATS Post Execute" << std::endl;
  std::cerr << "output: " << std::endl;
  output.PrintSummary(std::cerr);
  return output;
}

} // namespace vtkh
