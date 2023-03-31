#include "vtkmHistogram.hpp"
#include <vtkm/filter/density_estimate/Histogram.h>

namespace vtkh
{
vtkm::cont::PartitionedDataSet
vtkmHistogram::Run(vtkm::cont::PartitionedDataSet &p_input,
              vtkm::Id num_bins,
	      vtkm::Range range)
{
  vtkm::filter::density_estimate::Histogram hist;

  hist.SetNumberOfBins(num_bins);
  hist.SetRange(range);

  auto output = hist.Execute(p_input);
  return output;
}

} // namespace vtkh
