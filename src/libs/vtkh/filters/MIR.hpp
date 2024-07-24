#ifndef VTK_H_MIR_HPP
#define VTK_H_MIR_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/filters/Filter.hpp>
#include <memory>

namespace vtkh
{

class VTKH_API MIR: public Filter
{
public:
  MIR();
  virtual ~MIR();
  std::string GetName() const override;
  void SetMatSet(const std::string matset_name);
  void SetErrorScaling(const double error_scaling);
  void SetScalingDecay(const double scaling_decay);
  void SetIterations(const int iterations);
  void SetMaxError(const double max_error);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_matset_name;
  std::string m_lengths_name;
  std::string m_offsets_name;
  std::string m_ids_name;
  std::string m_vfs_name;
  double m_error_scaling;
  double m_scaling_decay;
  int m_iterations;
  double m_max_error;

};

} //namespace vtkh
#endif
