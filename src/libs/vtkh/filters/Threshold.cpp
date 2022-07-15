#include "Threshold.hpp"
#include <vtkh/Error.hpp>
#include <vtkh/filters/CleanGrid.hpp>
#include <vtkh/vtkm_filters/vtkmThreshold.hpp>

namespace vtkh
{

namespace detail
{

} // namespace detail

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

void
Threshold::SetAllInRange(const bool &value)
{
  m_return_all_in_range = value;
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

bool
Threshold::GetAllInRange() const
{
  return m_return_all_in_range;
}

std::string
Threshold::GetField() const
{
  return m_field_name;
}

void Threshold::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void Threshold::PostExecute()
{
  Filter::PostExecute();
}

void Threshold::DoExecute()
{

  DataSet temp_data;
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

    vtkmThreshold thresholder;

    auto data_set = thresholder.Run(dom,
                                    m_field_name,
                                    m_range.Min,
                                    m_range.Max,
                                    this->GetFieldSelection(),
                                    m_return_all_in_range);

    temp_data.AddDomain(data_set, domain_id);
  }

  CleanGrid cleaner;
  cleaner.SetInput(&temp_data);
  cleaner.Update();
  this->m_output = cleaner.GetOutput();

}

std::string
Threshold::GetName() const
{
  return "vtkh::Threshold";
}

} //  namespace vtkh
