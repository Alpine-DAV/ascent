#include <vtkh/filters/Tetrahedralize.hpp>
#include <vtkh/vtkm_filters/vtkmTetrahedralize.hpp>

namespace vtkh
{

Tetrahedralize::Tetrahedralize()
{

}

Tetrahedralize::~Tetrahedralize()
{

}

void Tetrahedralize::PreExecute()
{
  Filter::PreExecute();
}

void Tetrahedralize::PostExecute()
{
  Filter::PostExecute();
}

void Tetrahedralize::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    vtkmTetrahedralize tetter;
    // insert interesting stuff
    auto dataset = tetter.Run(dom, this->GetFieldSelection());

    m_output->AddDomain(dataset, domain_id);
  }
}

std::string
Tetrahedralize::GetName() const
{
  return "vtkh::Tetrahedralize";
}

} //  namespace vtkh
