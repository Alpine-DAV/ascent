#ifndef VTK_H_CONTOUR_TREE_HPP
#define VTK_H_CONTOUR_TREE_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>

namespace vtkh
{

class ContourTree : public Filter
{
public:
  ContourTree();
  virtual ~ContourTree();
  std::string GetName() const override;
  void SetField(const std::string &field_name);
  void SetNumLevels(int levels);
  std::vector<double> GetIsoValues();
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
  int m_levels;
  std::vector<double> m_iso_values;
};

} //namespace vtkh
#endif
