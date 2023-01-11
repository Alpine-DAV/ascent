#include "vtkmHistogram.hpp"
#include <vtkm/filter/density_estimate/Histogram.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmHistogram::Run(vtkm::cont::Field &f_input,
              vtkm::Id num_bins,
	      vtkm::Range range)
{
  vtkm::filter::density_estimate::Histogram hist;

  hist.SetNumberOfBins(num_bins);
  hist.SetRange(range);

  vtkm::cont::DataSet d_input;
  d_input.AddField(f_input);

  auto output = hist.Execute(d_input);
  return output;
}

} // namespace vtkh
