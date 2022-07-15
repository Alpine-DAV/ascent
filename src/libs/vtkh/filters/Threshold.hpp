#ifndef VTK_H_THRESHOLD_HPP
#define VTK_H_THRESHOLD_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/Range.h>

#include <memory>

namespace vtkh
{

class VTKH_API Threshold: public Filter
{
public:
  Threshold();
  virtual ~Threshold();
  std::string GetName() const override;
  void SetUpperThreshold(const double &value);
  void SetLowerThreshold(const double &value);
  void SetField(const std::string &field_name);
  void SetAllInRange(const bool &value);

  double GetUpperThreshold() const;
  double GetLowerThreshold() const;
  bool GetAllInRange() const;
  std::string GetField() const;
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  vtkm::Range m_range;
  std::string m_field_name;
  bool m_return_all_in_range = false;
};

} //namespace vtkh
#endif
