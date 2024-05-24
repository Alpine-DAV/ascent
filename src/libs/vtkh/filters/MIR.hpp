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
  void SetField(const std::string field_name);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
};

} //namespace vtkh
#endif
