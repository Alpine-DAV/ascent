#include <vtkh/filters/LinearExtrude.hpp>

#include <vtkh/viskores_filters/viskoresLinearExtrude.hpp>

namespace vtkh
{

LinearExtrude::LinearExtrude()
  : m_steps(1)
{
  m_vector[0] = 0.0;
  m_vector[1] = 0.0;
  m_vector[2] = 1.0;
}

LinearExtrude::~LinearExtrude()
{
}

std::string
LinearExtrude::GetName() const
{
  return "vtkh::LinearExtrude";
}

void
LinearExtrude::SetVector(const double vector[3])
{
  m_vector[0] = vector[0];
  m_vector[1] = vector[1];
  m_vector[2] = vector[2];
}

void
LinearExtrude::SetSteps(const int steps)
{
  m_steps = steps;
}

void
LinearExtrude::PreExecute()
{
  Filter::PreExecute();
}

void
LinearExtrude::PostExecute()
{
  Filter::PostExecute();
}

void
LinearExtrude::DoExecute()
{
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();
  for(int i = 0; i < num_domains; ++i)
  {
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    viskoresLinearExtrude extruder;
    auto dataset = extruder.Run(dom,
                                m_vector,
                                static_cast<viskores::Int32>(m_steps),
                                this->GetFieldSelection());
    m_output->AddDomain(dataset, domain_id);
  }
}

} // namespace vtkh

