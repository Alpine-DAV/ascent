#include "vtkmLog.hpp"

namespace vtkh
{
vtkm::cont::DataSet
vtkmLog::Run(vtkm::cont::DataSet &input,
              const vtkm::filter::field_transform::LogValues::LogBase log_base,
              vtkm::Float32 min_value)
{
  vtkm::filter::field_transform::LogValues logarithm;
  
  logarithm.SetBaseValue(log_base);
  logarithm.SetMinValue(min_value);

  auto output = logarithm.Execute(input);
  return output;
}

} // namespace vtkh
