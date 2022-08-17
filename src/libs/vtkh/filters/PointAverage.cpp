#include <vtkh/filters/PointAverage.hpp>
#include <vtkh/Error.hpp>
#include <vtkm/filter/PointAverage.h>

namespace vtkh
{

PointAverage::PointAverage()
{

}

PointAverage::~PointAverage()
{

}

void
PointAverage::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
PointAverage::SetOutputField(const std::string &field_name)
{
  m_output_field_name = field_name;
}

void PointAverage::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
  if(m_output_field_name == "")
  {
    throw Error("PointAverage: output field name not set");
  }
}

void PointAverage::PostExecute()
{
  Filter::PostExecute();
}

void PointAverage::DoExecute()
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

    vtkm::filter::PointAverage avg;
    avg.SetOutputFieldName(m_output_field_name);
    avg.SetActiveField(m_field_name);
    avg.SetFieldsToPass(this->GetFieldSelection());
    auto dataset = avg.Execute(dom);
    m_output->AddDomain(dataset, domain_id);
  }
}

std::string
PointAverage::GetName() const
{
  return "vtkh::PointAverage";
}

} //  namespace vtkh
