//#include <vtkm/filter/your_vtkm_filter.h>
#include <vtkh/filters/NoOp.hpp>

namespace vtkh
{

NoOp::NoOp()
{

}

NoOp::~NoOp()
{

}

void
NoOp::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void NoOp::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void NoOp::PostExecute()
{
  Filter::PostExecute();
}

void NoOp::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    // insert interesting stuff
    m_output->AddDomain(dom, domain_id);
  }
}

std::string
NoOp::GetName() const
{
  return "vtkh::NoOp";
}

} //  namespace vtkh
