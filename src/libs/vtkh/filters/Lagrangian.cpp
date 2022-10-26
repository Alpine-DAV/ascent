#include <iostream>
#include <vtkh/vtkm_filters/vtkmLagrangian.hpp>
#include <vtkh/filters/Lagrangian.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vtkm/filter/flow/Lagrangian.h>
#include <vtkm/Particle.h>

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

void 
Lagrangian::SetCycle(const int &cycle)
{
	m_cycle = cycle;
}

void
Lagrangian::SetBasisParticles(const vtkm::cont::ArrayHandle<vtkm::Particle> &basisParticles)
{
	m_basis_particles = basisParticles;
}

void
Lagrangian::SetBasisParticlesOriginal(const vtkm::cont::ArrayHandle<vtkm::Particle> &basisParticlesOriginal)
{
	m_basis_particles_original = basisParticlesOriginal;
}

void
Lagrangian::SetBasisParticleValidity(const vtkm::cont::ArrayHandle<vtkm::Id> &basisParticleValidity)
{
	m_basis_particle_validity = basisParticleValidity;
}

vtkm::cont::ArrayHandle<vtkm::Particle>
Lagrangian::GetBasisParticles()
{
	return m_basis_particles;
}

vtkm::cont::ArrayHandle<vtkm::Particle>
Lagrangian::GetBasisParticlesOriginal()
{
	return m_basis_particles_original;
}

vtkm::cont::ArrayHandle<vtkm::Id>
Lagrangian::GetBasisParticleValidity()
{
	return m_basis_particle_validity;
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
  int cycle = this->m_input->GetCycle();

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
                                                              m_cycle,
                                                              m_cust_res,
                                                              m_x_res,
                                                              m_y_res,
                                                              m_z_res,
							      m_basis_particles,
							      m_basis_particles_original,
							      m_basis_particle_validity);

    m_output->AddDomain(extractedBasis, domain_id);
  }
}

std::string
Lagrangian::GetName() const
{
  return "vtkh::Lagrangian";
}

} //  namespace vtkh
