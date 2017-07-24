#include <vtkh_threshold.hpp>
#include <vtkh_error.hpp>

#include <vtkm/filter/Threshold.h>

namespace vtkh 
{


Threshold::Threshold()
{
}

Threshold::~Threshold()
{

}

void 
Threshold::SetUpperThreshold(const double &value)
{
  m_range.Max = value;
}

void 
Threshold::SetLowerThreshold(const double &value)
{
  m_range.Min = value;
}

void 
Threshold::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

double 
Threshold::GetUpperThreshold() const
{
  return m_range.Max;
}

double 
Threshold::GetLowerThreshold() const
{
  return m_range.Min;
}

std::string
Threshold::GetField() const
{
  return m_field_name;
}

void Threshold::PreExecute() 
{

}

void Threshold::PostExecute()
{

}

void Threshold::DoExecute()
{
  
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains(); 

  vtkm::filter::Threshold thresholder;
  thresholder.SetUpperThreshold(m_range.Max);
  thresholder.SetLowerThreshold(m_range.Min);

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkm::filter::ResultDataSet res = thresholder.Execute(dom, m_field_name);

    for(size_t f = 0; f < m_map_fields.size(); ++f)
    {
      thresholder.MapFieldOntoOutput(res, dom.GetField(m_map_fields[f]));
    }
    this->m_output->AddDomain(res.GetDataSet(), domain_id);
    
  }
}

} //  namespace vtkh
