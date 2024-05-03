#include <vtkh/filters/Statistics.hpp>
#include <vtkh/vtkm_filters/vtkmStatistics.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkm/cont/Algorithm.h>
#include <vtkm/cont/ArrayHandleCast.h>
#include <vtkm/cont/Invoker.h>
#include <vtkm/cont/PartitionedDataSet.h>
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
Statistics::GetField() const
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
  this->m_output = new DataSet();

  if(!this->m_input->GlobalFieldExists(m_field_name))
  {
    throw Error("Statistics: field : '"+m_field_name+"' does not exist'");
  }

  std::vector<vtkm::cont::DataSet> vtkm_ds;

  
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    if(dom.HasField(m_field_name))
    {
      vtkm_ds.push_back(dom);
    }
  }

  vtkm::cont::PartitionedDataSet data_pds(vtkm_ds);
  vtkmStatistics stats;
  auto result = stats.Run(data_pds, m_field_name);

  int size = result.GetNumberOfFields();
  vtkm::cont::DataSet dom;
  
  for(int i = 0; i < size; i++)
  {
    //g_field will have assoc=Global which only goes with vtkm::PDS
    //convert to new field with assoc=WholeDataSet to put in vtkm::DS
    vtkm::cont::Field g_field = result.GetField(i);
    vtkm::cont::Field field(g_field.GetName(),vtkm::cont::Field::Association::WholeDataSet,g_field.GetData());
    dom.AddField(field);
  }
  this->m_output->AddDomain(dom,0);

  VTKH_DATA_CLOSE();
}

std::string
Statistics::GetName() const
{
  return "vtkh::Statistics";
}

} //  namespace vtkh
