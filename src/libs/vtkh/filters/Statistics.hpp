#ifndef VTK_H_STATISTICS_HPP
#define VTK_H_STATISTICS_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/Filter.hpp>

namespace vtkh
{

class VTKH_API Statistics: public Filter
{
public:
  Statistics();
  virtual ~Statistics();
  std::string GetName() const override;

  void SetField(const std::string &field_name);
  std::string GetField() const;
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;

};

} //namespace vtkh
#endif
