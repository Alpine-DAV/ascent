#include <vtkh/Error.hpp>
#include <vtkh/filters/VectorMagnitude.hpp>

#include <vtkh/viskores_filters/viskoresVectorMagnitude.hpp>
#include <viskores/filter/vector_analysis/VectorMagnitude.h>
#include <viskores/TypeList.h>

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
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkh::viskoresVectorMagnitude mag;
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
