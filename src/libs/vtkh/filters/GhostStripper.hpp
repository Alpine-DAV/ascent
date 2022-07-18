#ifndef VTK_H_GHOST_STRIPPER_HPP
#define VTK_H_GHOST_STRIPPER_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API GhostStripper : public Filter
{
public:
  GhostStripper();
  virtual ~GhostStripper();
  std::string GetName() const override;
  void SetField(const std::string &field_name);

  void SetMinValue(const vtkm::Int32 min);
  void SetMaxValue(const vtkm::Int32 min);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  vtkm::Int32 m_min_value;
  vtkm::Int32 m_max_value;
};

} //namespace vtkh
#endif
