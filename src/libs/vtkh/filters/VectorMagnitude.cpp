#include <vtkh/Error.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>

#include <vtkh/vtkm_filters/vtkmVectorMagnitude.hpp>
#include <vtkm/filter/vector_analysis/VectorMagnitude.h>
#include <vtkm/TypeList.h>

namespace vtkh
{

VectorMagnitude::VectorMagnitude()
{

}

VectorMagnitude::~VectorMagnitude()
{

}

void
VectorMagnitude::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
VectorMagnitude::SetResultName(const std::string name)
{
  m_out_name = name;
}

void VectorMagnitude::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);

  if(m_out_name == "")
  {
    m_out_name = m_field_name + "_magnitude";
  }
}

void VectorMagnitude::PostExecute()
{
  Filter::PostExecute();
}

void VectorMagnitude::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkh::vtkmVectorMagnitude mag;
    auto dataset = mag.Run(dom,
                           m_field_name,
                           m_out_name,
                           this->GetFieldSelection());

    m_output->AddDomain(dataset, domain_id);
  }
}

std::string
VectorMagnitude::GetName() const
{
  return "vtkh::VectorMagnitude";
}

} //  namespace vtkh
