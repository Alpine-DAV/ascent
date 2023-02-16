#include <vtkm/filter/clean_grid/CleanGrid.h>
#include <vtkh/Error.hpp>
#include <vtkh/filters/ParticleMerging.hpp>

namespace vtkh
{

ParticleMerging::ParticleMerging()
  : m_radius(-1)
{

}

ParticleMerging::~ParticleMerging()
{

}

void
ParticleMerging::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
ParticleMerging::SetRadius(const vtkm::Float64 radius)
{
  if(radius <= 0)
  {
    throw Error("Particle merging: radius must be greater than zero");
  }

  m_radius = radius;
}

void ParticleMerging::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
  if(!this->m_input->IsPointMesh())
  {
    throw Error("Particle Merging: input must be a point mesh");
  }
  if(m_radius == -1.)
  {
    throw Error("Particle merging: radius never set");
  }
}

void ParticleMerging::PostExecute()
{
  Filter::PostExecute();
}

void ParticleMerging::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    bool fast_merge = true;
    vtkm::filter::clean_grid::CleanGrid pointmerge;
    pointmerge.SetTolerance(m_radius * 2.);
    pointmerge.SetFastMerge(fast_merge);
    vtkm::cont::DataSet output = pointmerge.Execute(dom);

    m_output->AddDomain(output, domain_id);

  }
}

std::string
ParticleMerging::GetName() const
{
  return "vtkh::ParticleMerging";
}

} //  namespace vtkh
