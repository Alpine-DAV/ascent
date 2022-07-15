#ifndef VTK_H_PARTICLE_MERGING_HPP
#define VTK_H_PARTICLE_MERGING_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API ParticleMerging : public Filter
{
public:
  ParticleMerging();
  virtual ~ParticleMerging();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetRadius(const vtkm::Float64 radius);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  vtkm::Float64 m_radius;
};

} //namespace vtkh
#endif
