#include <vtkh/filters/Revolve.hpp>

#include <vtkh/viskores_filters/viskoresRevolve.hpp>

namespace vtkh
{

Revolve::Revolve()
  : m_start_angle_degrees(0.0),
    m_sweep_angle_degrees(360.0),
    m_steps(32),
    m_periodic(false)
{
  m_point[0] = 0.0;
  m_point[1] = 0.0;
  m_point[2] = 0.0;

  m_axis[0] = 0.0;
  m_axis[1] = 1.0;
  m_axis[2] = 0.0;
}

Revolve::~Revolve()
{
}

std::string
Revolve::GetName() const
{
  return "vtkh::Revolve";
}

void
Revolve::SetPoint(const double point[3])
{
  m_point[0] = point[0];
  m_point[1] = point[1];
  m_point[2] = point[2];
}

void
Revolve::SetAxis(const double axis[3])
{
  m_axis[0] = axis[0];
  m_axis[1] = axis[1];
  m_axis[2] = axis[2];
}

void
Revolve::SetStartAngle(const double start_angle_degrees)
{
  m_start_angle_degrees = start_angle_degrees;
}

void
Revolve::SetSweepAngle(const double sweep_angle_degrees)
{
  m_sweep_angle_degrees = sweep_angle_degrees;
}

void
Revolve::SetSteps(const int steps)
{
  m_steps = steps;
}

void
Revolve::SetPeriodic(const bool periodic)
{
  m_periodic = periodic;
}

void
Revolve::PreExecute()
{
  Filter::PreExecute();
}

void
Revolve::PostExecute()
{
  Filter::PostExecute();
}

void
Revolve::DoExecute()
{
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();
  for(int i = 0; i < num_domains; ++i)
  {
    viskores::Id domain_id;
    viskores::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    viskoresRevolve revolver;
    auto dataset = revolver.Run(dom,
                                m_point,
                                m_axis,
                                m_start_angle_degrees,
                                m_sweep_angle_degrees,
                                static_cast<viskores::Int32>(m_steps),
                                m_periodic,
                                this->GetFieldSelection());
    m_output->AddDomain(dataset, domain_id);
  }
}

} // namespace vtkh

