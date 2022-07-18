#include <vtkh/filters/Triangulate.hpp>
#include <vtkh/vtkm_filters/vtkmTriangulate.hpp>

namespace vtkh
{

Triangulate::Triangulate()
{

}

Triangulate::~Triangulate()
{

}

void Triangulate::PreExecute()
{
  Filter::PreExecute();
}

void Triangulate::PostExecute()
{
  Filter::PostExecute();
}

void Triangulate::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    vtkmTriangulate tetter;
    // insert interesting stuff
    auto dataset = tetter.Run(dom, this->GetFieldSelection());

    m_output->AddDomain(dataset, domain_id);
  }
}

std::string
Triangulate::GetName() const
{
  return "vtkh::Triangulate";
}

} //  namespace vtkh
