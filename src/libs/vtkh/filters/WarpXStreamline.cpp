#include <iostream>
#include <vtkh/vtkm_filters/vtkmWarpXStreamline.hpp>
#include <vtkh/filters/WarpXStreamline.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vtkm/filter/flow/WarpXStreamline.h>
#include <vtkm/Particle.h>

namespace vtkh
{

WarpXStreamline::WarpXStreamline()
{
}

WarpXStreamline::~WarpXStreamline()
{

}

void
WarpXStreamline::SetSteps(const double &steps)
{
  m_steps = steps;
}


void WarpXStreamline::PreExecute()
{
  Filter::PreExecute();
  Filter::CheckForRequiredField(m_field_name);
}

void WarpXStreamline::PostExecute()
{
  Filter::PostExecute();
}

void WarpXStreamline::DoExecute()
{
  vtkmWarpXStreamline warpXStreamlineFilter;

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
      auto field = dom.GetField(m_field_name).GetData();
      if(!field.IsType<vectorField_d>() && !field.IsType<vectorField_f>())
      {
        throw Error("Vector field type does not match <vtkm::Vec<vtkm::Float32,3>> or <vtkm::Vec<vtkm::Float64,3>>");
      }
    }
    else
    {
      throw Error("Domain does not contain specified vector field for WarpXStreamline analysis.");
    }

    vtkm::cont::DataSet output = warpXStreamlineFilter.Run(dom,
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

    m_output->AddDomain(output, domain_id);
  }
}

std::string
WarpXStreamline::GetName() const
{
  return "vtkh::WarpXStreamline";
}

} //  namespace vtkh
