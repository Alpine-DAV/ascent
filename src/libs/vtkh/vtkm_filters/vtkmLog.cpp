#include "vtkmLog.hpp"

namespace vtkh
{
vtkm::cont::DataSet
vtkmLog::Run(vtkm::cont::DataSet &input,
	     const std::string in_field_name,
	     const std::string out_field_name,
	     vtkm::cont::Field::Association in_assoc,
             int log_base,
             vtkm::Float32 min_value)
{
  typedef vtkm::filter::field_transform::LogValues vtkmLogFilter;
  vtkmLogFilter logarithm;
  
  if(log_base == 1) logarithm.SetBaseValue(vtkmLogFilter::LogBase::E);
  if(log_base == 2) logarithm.SetBaseValue(vtkmLogFilter::LogBase::TWO);
  if(log_base == 10) logarithm.SetBaseValue(vtkmLogFilter::LogBase::TEN);

  logarithm.SetActiveField(in_field_name, in_assoc);
  logarithm.SetOutputFieldName(out_field_name);
  logarithm.SetMinValue(min_value);
  auto output = logarithm.Execute(input);
  return output;
}

} // namespace vtkh
