#ifndef VTK_H_LAGRANGIAN_HPP
#define VTK_H_LAGRANGIAN_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <viskores/filter/flow/Lagrangian.h>
#include <viskores/Particle.h>

namespace vtkh
{

class VTKH_API Lagrangian : public Filter
{
public:
  Lagrangian();
  virtual ~Lagrangian();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetCycle(const int &cycle);
  void SetStepSize(const double &step_size);
  void SetWriteFrequency(const int &write_frequency);
  void SetCustomSeedResolution(const int &cust_res);
  void SetSeedResolutionInX(const int &x_res);
  void SetSeedResolutionInY(const int &y_res);
  void SetSeedResolutionInZ(const int &z_res);
  void SetBasisParticles(const viskores::cont::ArrayHandle<viskores::Particle> &basisParticles);
  void SetBasisParticlesOriginal(const viskores::cont::ArrayHandle<viskores::Particle> &basisParticlesOriginal);
  void SetBasisParticleValidity(const viskores::cont::ArrayHandle<viskores::Id> &basisParticleValidity);
  viskores::cont::ArrayHandle<viskores::Particle> GetBasisParticles();
  viskores::cont::ArrayHandle<viskores::Particle> GetBasisParticlesOriginal();
  viskores::cont::ArrayHandle<viskores::Id> GetBasisParticleValidity();


protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  double m_step_size;
  int m_write_frequency;
  int m_cycle;
  int m_cust_res;
  int m_x_res, m_y_res, m_z_res;
  viskores::cont::ArrayHandle<viskores::Particle> m_basis_particles;
  viskores::cont::ArrayHandle<viskores::Particle> m_basis_particles_original;
  viskores::cont::ArrayHandle<viskores::Id> m_basis_particle_validity;
};

} //namespace vtkh
#endif
