#include <vtkh/filters/CellAverage.hpp>
#include <vtkh/vtkm_filters/vtkmCellAverage.hpp>
#include <vtkh/Error.hpp>

namespace vtkh
{

CellAverage::CellAverage()
{

}

CellAverage::~CellAverage()
{

}

void
CellAverage::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
CellAverage::SetOutputField(const std::string &field_name)
{
  m_output_field_name = field_name;
}

void CellAverage::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);

  if(m_output_field_name == "")
  {
    throw Error("CellAverage: output field name not set");
  }
}

void CellAverage::PostExecute()
{
  Filter::PostExecute();
}

void CellAverage::DoExecute()
{
  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    if(!dom.HasField(m_field_name))
    {
      continue;
    }

    vtkh::vtkmCellAverage avg;
    auto dataset = avg.Run(dom,
                           m_field_name,
                           m_output_field_name,
                           this->GetFieldSelection());
    m_output->AddDomain(dataset, domain_id);
  }
}

std::string
CellAverage::GetName() const
{
  return "vtkh::CellAverage";
}

} //  namespace vtkh
