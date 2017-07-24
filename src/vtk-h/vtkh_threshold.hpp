#ifndef VTK_H_THRESHOLD_HPP
#define VTK_H_THRESHOLD_HPP

#include "vtkh.hpp"
#include "vtkh_filter.hpp"
#include "vtkh_data_set.hpp"
#include <vtkm/Range.h>

#include <memory>

namespace vtkh
{

class Threshold: public Filter
{
public:
  Threshold(); 
  virtual ~Threshold(); 
  void SetUpperThreshold(const double &value);
  void SetLowerThreshold(const double &value);
  void SetField(const std::string &field_name);

  double GetUpperThreshold() const;
  double GetLowerThreshold() const;
  std::string GetField() const;
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  vtkm::Range m_range;
  std::string m_field_name;
};

} //namespace vtkh
#endif
