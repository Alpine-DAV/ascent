#ifndef VTK_H_WARPX_STREAMLINE_HPP
#define VTK_H_WARPX_STREAMLINE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

#include <vtkm/Particle.h>

namespace vtkh
{

class VTKH_API WarpXStreamline : public Filter
{
public:
  WarpXStreamline();
  virtual ~WarpXStreamline();
  std::string GetName() const override { return "vtkh::WarpXStreamline";}
  void SetBField(const std::string &field_name) {  m_b_field_name = field_name; }
  void SetEField(const std::string &field_name) {  m_e_field_name = field_name; }
  void SetChargeField(const std::string &field_name) {  m_charge_field_name = field_name; }
  void SetMassField(const std::string &field_name) {  m_mass_field_name = field_name; }
  void SetMomentumField(const std::string &field_name) {  m_momentum_field_name = field_name; }
  void SetWeightingField(const std::string &field_name) {  m_weighting_field_name = field_name; }
  void SetStepSize(const double &step_size) {   m_step_size = step_size; }
  void SetNumberOfSteps(int numSteps) { m_num_steps = numSteps; }
  void SetTubes(bool tubes) {m_tubes = tubes;}
  void SetTubeCapping(bool capping) {m_tube_capping = capping;}
  void SetTubeValue(double val) {m_tube_value = val;}
  void SetTubeSize(double size) {m_tube_size = size; m_radius_set = true;}
  void SetTubeSides(double sides) {m_tube_sides = sides;}
  void SetOutputField(const std::string &output_field_name) {  m_output_field_name = output_field_name; }

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_b_field_name;
  std::string m_e_field_name;
  std::string m_charge_field_name;
  std::string m_mass_field_name;
  std::string m_momentum_field_name;
  std::string m_weighting_field_name;
  std::string m_output_field_name;
  bool m_tubes;
  bool m_tube_capping;
  bool m_radius_set;
  double m_tube_value;
  double m_tube_size;
  double m_tube_sides;
  double m_step_size;
  int m_num_steps;
};

} //namespace vtkh
#endif
