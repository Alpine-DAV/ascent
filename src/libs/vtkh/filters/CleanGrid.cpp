
#include <vtkh/filters/CleanGrid.hpp>
#include <vtkh/Error.hpp>

#include <vtkh/vtkm_filters/vtkmCleanGrid.hpp>

namespace vtkh
{


CleanGrid::CleanGrid()
  : m_tolerance(-1.)
{

}

CleanGrid::~CleanGrid()
{

}

void
CleanGrid::PreExecute()
{
  Filter::PreExecute();
}

void
CleanGrid::DoExecute()
{
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkh::vtkmCleanGrid cleaner;
    if(m_tolerance != -1.)
    {
      cleaner.tolerance(m_tolerance);
    }
    auto dataset = cleaner.Run(dom, this->GetFieldSelection());
    this->m_output->AddDomain(dataset, domain_id);
  }

}

void
CleanGrid::PostExecute()
{
  Filter::PostExecute();
}

std::string
CleanGrid::GetName() const
{
  return "vtkh::CleanGrid";
}

void
CleanGrid::Tolerance(const vtkm::Float64 tolerance)
{
  m_tolerance = tolerance;
}

} // namespace vtkh
