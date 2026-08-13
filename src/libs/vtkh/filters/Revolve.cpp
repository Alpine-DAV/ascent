#include <vtkh/filters/Revolve.hpp>
#include <vtkh/viskores_filters/viskoresRevolve.hpp>

namespace vtkh
{

Revolve::Revolve()
  : m_axis(1.f, 0.f, 0.f),
    m_point(0.f, 0.f, 0.f),
    m_angle_degrees(360.f),
    m_num_steps(16),
    m_capping(false)
{
}

Revolve::~Revolve() = default;

void Revolve::SetAxis(const viskores::Vec3f &axis) { m_axis = axis; }
void Revolve::SetPoint(const viskores::Vec3f &point) { m_point = point; }
void Revolve::SetAngleDegrees(viskores::FloatDefault angle_degrees) { m_angle_degrees = angle_degrees; }
void Revolve::SetNumSteps(viskores::Id num_steps) { m_num_steps = num_steps; }
void Revolve::SetCapping(bool capping) { m_capping = capping; }

void Revolve::PreExecute()
{
  Filter::PreExecute();
}

void Revolve::PostExecute()
{
  Filter::PostExecute();
}

void Revolve::DoExecute()
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
                                this->GetFieldSelection(),
                                this->m_axis,
                                this->m_point,
                                this->m_angle_degrees,
                                this->m_num_steps,
                                this->m_capping);

    m_output->AddDomain(dataset, domain_id);
  }
}

std::string Revolve::GetName() const
{
  return "vtkh::Revolve";
}

} // namespace vtkh

