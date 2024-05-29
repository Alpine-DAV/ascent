#include "MIR.hpp"

#include <vtkh/vtkm_filters/vtkmMIR.hpp>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/cont/Invoker.h>

namespace vtkh
{

namespace detail
{

bool
isMaterial(std::string matset_name, std::string field_name)
{

  // Create the substring to search for
  std::string searchString = matset_name + "_";

  // Check if the fieldName contains the searchString
  if (field_name.find(searchString) != std::string::npos) {
      return true;
  }
  return false;
}

class MetaDataLength : public vtkm::worklet::WorkletMapField
{
public:
  using ControlSignature = void(FieldIn, FieldOut);

  VTKM_EXEC
  void operator()(const vtkm::FloatDefault& vf_data,
                  vtkm::Id& length) const
  {
    if (vf_data > vtkm::FloatDefault(0.0))
    {
      length = length + 1;
      //std::cerr << "length before: " << length << std::endl;
      //length++;
      //std::cerr << "length after: " << length << std::endl;
      //std::cerr << std::endl;
    }
  }
};


}//end detail

MIR::MIR()
{

}

MIR::~MIR()
{

}

void
MIR::SetMatSet(const std::string matset_name)
{
  m_matset_name = matset_name;
}

void
MIR::PreExecute()
{
  Filter::PreExecute();
  //Filter::CheckForRequiredField(m_field_name);
}

void
MIR::PostExecute()
{
  Filter::PostExecute();
}

void MIR::DoExecute()
{
  vtkm::cont::Invoker invoker;
  const int num_domains = this->m_input->GetNumberOfDomains();
  vtkm::cont::ArrayHandle<vtkm::Id> length;
  
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    vtkm::Id num_fields = dom.GetNumberOfFields();
    for(int j = 0; j < num_fields; ++j)
    {
      vtkm::cont::Field field = dom.GetField(j);
      std::string field_name = field.GetName();
      bool is_material = detail::isMaterial(m_matset_name, field_name);
      std::cerr << "isMaterial( " << field_name << " ): " << is_material << std::endl;
      if(is_material)
      {
        vtkm::cont::ArrayHandle<vtkm::FloatDefault> data;
        field.GetDataAsDefaultFloat().AsArrayHandle(data);
        vtkm::Id num_values = data.GetNumberOfValues();
        if(length.GetNumberOfValues() != num_values)
        {
          std::cerr << "HERE" << std::endl;
          length.AllocateAndFill(num_values,0.0);
        }
        invoker(detail::MetaDataLength{}, data, length);
        std::cerr << "length now: " << std::endl;
        for(int n = 0; n < num_values; ++n)
        {
          std::cerr << length.ReadPortal().Get(n) << " ";
        }
      }
    }
  }
}

std::string
MIR::GetName() const
{
  return "vtkh::MIR";
}

} //  namespace vtkh
