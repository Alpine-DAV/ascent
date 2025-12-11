#include <vtkh/filters/MeshQuality.hpp>
#include <vtkh/viskores_filters/viskoresMeshQuality.hpp>
#include <vtkh/viskores_filters/viskoresCleanGrid.hpp>
#include <vtkh/Error.hpp>

namespace vtkh
{

MeshQuality::MeshQuality()
  : m_metric(viskores::filter::mesh_info::CellMetric::Volume)
{

}

MeshQuality::~MeshQuality()
{

}

void MeshQuality::cell_metric(viskores::filter::mesh_info::CellMetric metric)
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
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    // force this to an fully explicit data set because
    // old viskores was not handling this
    vtkh::viskoresCleanGrid cleaner;
    auto dataset = cleaner.Run(dom, this->GetFieldSelection());

    viskoresMeshQuality quali;
    viskores::cont::DataSet res = quali.Run(dataset, m_metric, this->GetFieldSelection());
    m_output->AddDomain(res, domain_id);
  }
}

std::string
MeshQuality::GetName() const
{
  return "vtkh::MeshQuality";
}

} //  namespace vtkh
