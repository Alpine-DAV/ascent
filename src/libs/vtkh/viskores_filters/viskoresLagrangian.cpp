#include "viskoresLagrangian.hpp"

#include <viskores/filter/flow/Lagrangian.h>
#include <viskores/Particle.h>

namespace vtkh
{
viskores::cont::DataSet
viskoresLagrangian::Run(viskores::cont::DataSet &input,
                         std::string field_name,
                         double step_size,
                         int write_frequency,
                         int cycle,
                         int cust_res,
                         int x_res,
                         int y_res,
                         int z_res,
			 viskores::cont::ArrayHandle<viskores::Particle> basis_particles,
			 viskores::cont::ArrayHandle<viskores::Particle> basis_particles_original,
			 viskores::cont::ArrayHandle<viskores::Id> basis_particle_validity)
{
#ifdef VTKH_BYPASS_VISKORES_BIH
  return viskores::cont::DataSet();
#else
  viskores::filter::flow::Lagrangian lagrangianFilter;

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
