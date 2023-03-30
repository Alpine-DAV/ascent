#include "vtkmCompositeVector.hpp"

namespace vtkh
{
vtkm::cont::DataSet
vtkmCompositeVector::Run(vtkm::cont::DataSet &input,
	     std::vector<std::string> input_field_names,
	     const std::string output_field_name,
	     vtkm::cont::Field::Association assoc)
{
  vtkm::filter::field_transform::CompositeVectors compvec;
  
  compvec.SetFieldNameList(input_field_names, assoc);
  compvec.SetOutputFieldName(output_field_name);
  
  auto output = compvec.Execute(input);
  
  return output;
}

} // namespace vtkh
