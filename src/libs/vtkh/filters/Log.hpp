#ifndef VTK_H_LOG_HPP
#define VTK_H_LOG_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/Range.h>

namespace vtkh
{

class VTKH_API Log: public Filter
{
public:
  Log();
  virtual ~Log();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetResultField(const std::string &field_name);
  void SetClampToMin(bool on);
  void SetClampMin(vtkm::Float32 min_value);

  std::string GetField() const;
  std::string GetResultField() const;
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::string m_field_name;
  std::string m_result_name;
  vtkm::Float32 m_min_value;
  bool m_clamp_to_min;

};


class VTKH_API Log10: public Filter
{
public:
  Log10();
  virtual ~Log10();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetResultField(const std::string &field_name);
  void SetClampToMin(bool on);
  void SetClampMin(vtkm::Float32 min_value);

  std::string GetField() const;
  std::string GetResultField() const;
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::string m_field_name;
  std::string m_result_name;
  vtkm::Float32 m_min_value;
  bool m_clamp_to_min;

};

class VTKH_API Log2: public Filter
{
public:
  Log2();
  virtual ~Log2();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetResultField(const std::string &field_name);
  void SetClampToMin(bool on);
  void SetClampMin(vtkm::Float32 min_value);

  std::string GetField() const;
  std::string GetResultField() const;
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::string m_field_name;
  std::string m_result_name;
  vtkm::Float32 m_min_value;
  bool m_clamp_to_min;

};

} //namespace vtkh
#endif
