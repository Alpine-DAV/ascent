#include "viskoresHistogram.hpp"
#include <viskores/filter/density_estimate/Histogram.h>

namespace vtkh
{
viskores::cont::PartitionedDataSet
viskoresHistogram::Run(viskores::cont::PartitionedDataSet &p_input,
              viskores::Id num_bins,
	      viskores::Range range)
{
  viskores::filter::density_estimate::Histogram hist;

  hist.SetNumberOfBins(num_bins);
  hist.SetRange(range);

  auto output = hist.Execute(p_input);
  return output;
}

} // namespace vtkh
