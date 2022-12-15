#ifndef VTK_H_SLICE_HPP
#define VTK_H_SLICE_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>


namespace vtkh
{

class VTKH_API Slice : public Filter
{
public:
  Slice();
  virtual ~Slice();
  std::string GetName() const override;
  void AddPlane(vtkm::Vec<vtkm::Float32,3> point, vtkm::Vec<vtkm::Float32,3> normal);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::vector<vtkm::Vec<vtkm::Float32,3>> m_points;
  std::vector<vtkm::Vec<vtkm::Float32,3>> m_normals;
};

class VTKH_API AutoSliceLevels : public Filter
{
public:
  AutoSliceLevels();
  virtual ~AutoSliceLevels();
  std::string GetName() const override;
  void SetNormal(vtkm::Vec<vtkm::Float32,3> normal);
  void SetLevels(int levels);
  void SetField(std::string field_name);
protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  std::vector<vtkm::Vec<vtkm::Float32,3>> m_normals;
  int m_levels;
  std::string m_field_name;
};

} //namespace vtkh
#endif
