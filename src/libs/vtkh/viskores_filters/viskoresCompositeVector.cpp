#include "viskoresCompositeVector.hpp"

namespace vtkh
{
viskores::cont::DataSet
viskoresCompositeVector::Run(viskores::cont::DataSet &input,
	     std::vector<std::string> input_field_names,
	     const std::string output_field_name,
	     viskores::cont::Field::Association assoc)
{
  viskores::filter::field_transform::CompositeVectors compvec;
  
  compvec.SetFieldNameList(input_field_names, assoc);
  compvec.SetOutputFieldName(output_field_name);
  
  auto output = compvec.Execute(input);
  
  return output;
}

} // namespace vtkh
