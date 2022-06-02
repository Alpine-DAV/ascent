#ifndef VTK_H_CELL_AVERAGE_HPP
#define VTK_H_CELL_AVERAGE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class VTKH_API CellAverage : public Filter
{
public:
  CellAverage();
  virtual ~CellAverage();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetOutputField(const std::string &field_name);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name, m_output_field_name;
};

} //namespace vtkh
#endif
