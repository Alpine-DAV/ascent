#include "vtkmLagrangian.hpp"

#include <vtkm/filter/flow/Lagrangian.h>
#include <vtkm/Particle.h>

namespace vtkh
{
vtkm::cont::DataSet
vtkmLagrangian::Run(vtkm::cont::DataSet &input,
                         std::string field_name,
                         double step_size,
                         int write_frequency,
                         int cycle,
                         int cust_res,
                         int x_res,
                         int y_res,
                         int z_res,
			 vtkm::cont::ArrayHandle<vtkm::Particle> basis_particles,
			 vtkm::cont::ArrayHandle<vtkm::Particle> basis_particles_original,
			 vtkm::cont::ArrayHandle<vtkm::Id> basis_particle_validity)
{
#ifdef VTKH_BYPASS_VTKM_BIH
  return vtkm::cont::DataSet();
#else
  vtkm::filter::flow::Lagrangian lagrangianFilter;

  lagrangianFilter.SetStepSize(step_size);
  lagrangianFilter.SetCycle(cycle);
  lagrangianFilter.SetWriteFrequency(write_frequency);
  lagrangianFilter.SetActiveField(field_name);
  lagrangianFilter.SetCustomSeedResolution(cust_res);
  lagrangianFilter.SetSeedResolutionInX(x_res);
  lagrangianFilter.SetSeedResolutionInY(y_res);
  lagrangianFilter.SetSeedResolutionInZ(z_res);
  lagrangianFilter.SetBasisParticles(basis_particles);
  lagrangianFilter.SetBasisParticlesOriginal(basis_particles_original);
  lagrangianFilter.SetBasisParticleValidity(basis_particle_validity);

  auto output = lagrangianFilter.Execute(input);

  return output;
#endif
}

} // namespace vtkh
