#include <vtkh/filters/Statistics.hpp>
#include <vtkh/viskores_filters/viskoresStatistics.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleCast.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/PartitionedDataSet.h>
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

  std::vector<viskores::cont::DataSet> viskores_ds;

  
  for(int i = 0; i < num_domains; ++i)
  {
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    if(dom.HasField(m_field_name))
    {
      viskores_ds.push_back(dom);
    }
  }

  viskores::cont::PartitionedDataSet data_pds(viskores_ds);
  viskoresStatistics stats;
  auto result = stats.Run(data_pds, m_field_name);

  int size = result.GetNumberOfFields();
  viskores::cont::DataSet dom;
  
  for(int i = 0; i < size; i++)
  {
    //g_field will have assoc=Global which only goes with viskores::PDS
    //convert to new field with assoc=WholeDataSet to put in viskores::DS
    viskores::cont::Field g_field = result.GetField(i);
    viskores::cont::Field field(g_field.GetName(),viskores::cont::Field::Association::WholeDataSet,g_field.GetData());
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
