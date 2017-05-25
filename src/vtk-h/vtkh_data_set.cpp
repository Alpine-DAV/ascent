#include<vtkh_data_set.hpp>

namespace vtkh {

vtkm::Bounds 
vtkhDataSet::GetBounds() const
{

}

vtkm::cont::ArrayHandle<vtkm::Range> 
vtkhDataSet::GetRange(const std::string &field_name) const
{
  bool valid_field = true;
  const int num_domains;
  
  for(size_t i = 0; i <

}

} // namspace vtkh
