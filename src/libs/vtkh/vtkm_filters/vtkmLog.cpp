#include "vtkmLog.hpp"

namespace vtkh
{
vtkm::cont::DataSet
vtkmLog::Run(vtkm::cont::DataSet &input,
	     const std::string in_field_name,
	     const std::string out_field_name,
	     vtkm::cont::Field::Association in_assoc,
             vtkmLogFilter::LogBase log_base,
             vtkm::Float32 min_value)
{
  vtkmLogFilter logarithm;
  
  logarithm.SetActiveField(in_field_name, in_assoc);
  logarithm.SetOutputFieldName(out_field_name);
  logarithm.SetBaseValue(log_base);
  logarithm.SetMinValue(min_value);
  
  auto output = logarithm.Execute(input);
  
  return output;
}

} // namespace vtkh
