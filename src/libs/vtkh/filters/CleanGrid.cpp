
#include <vtkh/filters/CleanGrid.hpp>
#include <vtkh/Error.hpp>

#include <vtkh/viskores_filters/viskoresCleanGrid.hpp>

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
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkh::viskoresCleanGrid cleaner;
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
CleanGrid::Tolerance(const viskores::Float64 tolerance)
{
  m_tolerance = tolerance;
}

} // namespace vtkh
