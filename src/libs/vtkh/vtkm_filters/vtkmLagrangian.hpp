#ifndef VTK_H_VISKORES_LAGRANGIAN_HPP
#define VTK_H_VISKORES_LAGRANGIAN_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/Particle.h>

namespace vtkh
{

class viskoresLagrangian
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
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
			  viskores::cont::ArrayHandle<viskores::Id> basis_particle_validity);
};
}
#endif
