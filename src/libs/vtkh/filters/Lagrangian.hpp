#ifndef VTK_H_LAGRANGIAN_HPP
#define VTK_H_LAGRANGIAN_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/filter/flow/Lagrangian.h>
#include <vtkm/Particle.h>

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
  void SetBasisParticles(const vtkm::cont::ArrayHandle<vtkm::Particle> &basisParticles);
  void SetBasisParticlesOriginal(const vtkm::cont::ArrayHandle<vtkm::Particle> &basisParticlesOriginal);
  void SetBasisParticleValidity(const vtkm::cont::ArrayHandle<vtkm::Id> &basisParticleValidity);
  vtkm::cont::ArrayHandle<vtkm::Particle> GetBasisParticles();
  vtkm::cont::ArrayHandle<vtkm::Particle> GetBasisParticlesOriginal();
  vtkm::cont::ArrayHandle<vtkm::Id> GetBasisParticleValidity();


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
  vtkm::cont::ArrayHandle<vtkm::Particle> m_basis_particles;
  vtkm::cont::ArrayHandle<vtkm::Particle> m_basis_particles_original;
  vtkm::cont::ArrayHandle<vtkm::Id> m_basis_particle_validity;
};

} //namespace vtkh
#endif
