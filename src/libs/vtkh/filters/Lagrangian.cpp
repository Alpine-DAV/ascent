#include <iostream>
#include <vtkh/vtkm_filters/vtkmLagrangian.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>

namespace vtkh
{

Lagrangian::Lagrangian()
{
}

Lagrangian::~Lagrangian()
{

}

void
Lagrangian::SetField(const std::string &field_name)
{
  m_field_name = field_name;
}

void
Lagrangian::SetStepSize(const double &step_size)
{
  m_step_size = step_size;
}

void
Lagrangian::SetWriteFrequency(const int &write_frequency)
{
  m_write_frequency = write_frequency;
}

void
Lagrangian::SetCustomSeedResolution(const int &cust_res)
{
	m_cust_res = cust_res;
}

void
Lagrangian::SetSeedResolutionInX(const int &x_res)
{
	m_x_res = x_res;
}

void
Lagrangian::SetSeedResolutionInY(const int &y_res)
{
	m_y_res = y_res;
}
void
Lagrangian::SetSeedResolutionInZ(const int &z_res)
{
	m_z_res = z_res;
}


void Lagrangian::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void Lagrangian::PostExecute()
{
  Filter::PostExecute();
}

void Lagrangian::DoExecute()
{
  vtkmLagrangian lagrangianFilter;

  this->m_output = new DataSet();
  const int num_domains = this->m_input->GetNumberOfDomains();
  for(int i = 0; i < num_domains; ++i)
  {
    vtkm::Id domain_id;
    vtkm::cont::DataSet dom;
    this->m_input->GetDomain(i, dom, domain_id);
    if(dom.HasField(m_field_name))
    {
      using vectorField_d = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float64, 3>>;
      using vectorField_f = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::Float32, 3>>;
      auto field = dom.GetField(m_field_name).GetData();
      if(!field.IsType<vectorField_d>() && !field.IsType<vectorField_f>())
      {
        throw Error("Vector field type does not match <vtkm::Vec<vtkm::Float32,3>> or <vtkm::Vec<vtkm::Float64,3>>");
      }
    }
    else
    {
      throw Error("Domain does not contain specified vector field for Lagrangian analysis.");
    }

    vtkm::cont::DataSet extractedBasis = lagrangianFilter.Run(dom,
                                                              m_field_name,
                                                              m_step_size,
                                                              m_write_frequency,
                                                              vtkh::GetMPIRank(),
                                                              m_cust_res,
                                                              m_x_res,
                                                              m_y_res,
                                                              m_z_res);

    m_output->AddDomain(extractedBasis, domain_id);
  }
}

std::string
Lagrangian::GetName() const
{
  return "vtkh::Lagrangian";
}

} //  namespace vtkh
