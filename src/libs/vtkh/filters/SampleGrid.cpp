
#include <vtkh/filters/SampleGrid.hpp>
#include <vtkh/Error.hpp>

#include <vtkh/vtkm_filters/vtkmProbe.hpp>
#include <limits>

namespace vtkh
{

SampleGrid::SampleGrid()
	: m_invalid_value(std::numeric_limits<double>::min())
{

}

SampleGrid::~SampleGrid()
{

}

void
SampleGrid::PreExecute()
{
  Filter::PreExecute();
}

void
SampleGrid::DoExecute()
{
  this->m_output = new DataSet();

  const int num_domains = this->m_input->GetNumberOfDomains();

  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);

    vtkh::vtkmProbe probe;
    probe.dims(m_dims);
    probe.origin(m_origin);
    probe.spacing(m_spacing);
    probe.invalidValue(m_invalid_value);

    auto dataset = probe.Run(dom);
    this->m_output->AddDomain(dataset, domain_id);
  }

}

void
SampleGrid::PostExecute()
{
  Filter::PostExecute();
}

std::string
SampleGrid::GetName() const
{
  return "vtkh::SampleGrid";
}

void
SampleGrid::Dims(const Vec3f dims)
{
  m_dims = dims;
}

void
SampleGrid::Origin(const Vec3f origin)
{
  m_origin = origin;
}

void
SampleGrid::Spacing(const Vec3f spacing)
{
  m_spacing = spacing;
}

void
SampleGrid::InvalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

} // namespace vtkh