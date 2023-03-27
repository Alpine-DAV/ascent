#include <vtkh/filters/Statistics.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/Invoker.h>
#include <vector>

#ifdef VTKH_PARALLEL
#include <mpi.h>
#endif

namespace vtkh
{

namespace detail
{

} // namespace detail

Statistics::Statistics()
{

}

Statistics::~Statistics()
{

}

void
Statistics::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

std::string
Statistics::GetField()
{
  return m_field_name;
}

void
Statistics::PreExecute()
{
  Filter::PreExecute();
}

void
Statistics::PostExecute()
{
  Filter::PostExecute();
}

void Statistics::DoExecute()
{
  VTKH_DATA_OPEN("statistics");
  VTKH_DATA_ADD("device", GetCurrentDevice());
  VTKH_DATA_ADD("input_cells", this->m_input->GetNumberOfCells());
  VTKH_DATA_ADD("input_domains", this->m_input->GetNumberOfDomains());
  const int num_domains = this->m_input->GetNumberOfDomains();

  if(!this->m_input->GlobalFieldExists(m_field_name))
  {
    throw Error("Statistics: field : '"+m_field_name+"' does not exist'");
  }

  vtkm::cont::PartitionedDataSet data_pds;
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    data_set.GetDomain(i, dom, domain_id);
    if(dom.HasField(field_name))
    {
      data_pds.AddParititon(dom);
    }
  }

  vtkmStatistics stats;
  auto result = stats.Run(data_pds, m_field_name);

  std::vector<vtkm::cont::DataSet> v_datasets = result.GetPartitions();
  int size = v_datasets.size();
  for(int i = 0; i < size; i++)
    this->m_output->AddDomain(v_datasets[i],i);


  VTKH_DATA_CLOSE();
}

std::string
Statistics::GetName() const
{
  return "vtkh::Statistics";
}

} //  namespace vtkh
