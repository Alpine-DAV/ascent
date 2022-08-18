#include <vtkh/filters/MeshQuality.hpp>
#include <vtkh/vtkm_filters/vtkmMeshQuality.hpp>
#include <vtkh/vtkm_filters/vtkmCleanGrid.hpp>
#include <vtkh/Error.hpp>

namespace vtkh
{

MeshQuality::MeshQuality()
  : m_metric(vtkm::filter::mesh_info::CellMetric::Volume)
{

}

MeshQuality::~MeshQuality()
{

}

void MeshQuality::cell_metric(vtkm::filter::mesh_info::CellMetric metric)
{
  m_metric = metric;
}

void MeshQuality::PreExecute()
{
  Filter::PreExecute();
  if(!m_input->IsUnstructured())
  {
    throw Error("Mesh quality requires that meshes be completely unstructured");
  }
}

void MeshQuality::PostExecute()
{
  Filter::PostExecute();
}

void MeshQuality::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    // force this to an fully explicit data set because
    // old vtkm was not handling this
    vtkh::vtkmCleanGrid cleaner;
    auto dataset = cleaner.Run(dom, this->GetFieldSelection());

    vtkmMeshQuality quali;
    vtkm::cont::DataSet res = quali.Run(dataset, m_metric, this->GetFieldSelection());
    m_output->AddDomain(res, domain_id);
  }
}

std::string
MeshQuality::GetName() const
{
  return "vtkh::MeshQuality";
}

} //  namespace vtkh
