#include "viskoresLog.hpp"

namespace vtkh
{
viskores::cont::DataSet
viskoresLog::Run(viskores::cont::DataSet &input,
	     const std::string in_field_name,
	     const std::string out_field_name,
	     viskores::cont::Field::Association in_assoc,
             viskoresLogFilter::LogBase log_base,
             viskores::Float32 min_value)
{
  viskoresLogFilter logarithm;
  
  logarithm.SetActiveField(in_field_name, in_assoc);
  logarithm.SetOutputFieldName(out_field_name);
  logarithm.SetBaseValue(log_base);
  logarithm.SetMinValue(min_value);
  
  auto output = logarithm.Execute(input);
  
  return output;
}

} // namespace vtkh
