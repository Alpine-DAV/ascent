#ifndef VTK_H_MESH_QUALITY_HPP
#define VTK_H_MESH_QUALITY_HPP

#include <vtkh/vtkh_exports.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/filter/MeshQuality.h>

namespace vtkh
{

class VTKH_API MeshQuality: public Filter
{
public:
  MeshQuality();
  virtual ~MeshQuality();
  std::string GetName() const override;

  void cell_metric(vtkm::filter::CellMetric metric);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;
  vtkm::filter::CellMetric m_metric;
};

} //namespace vtkh
#endif
