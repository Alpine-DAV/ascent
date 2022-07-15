#ifndef VTK_H_VTKM_MESH_QUALITY_HPP
#define VTK_H_VTKM_MESH_QUALITY_HPP

#include <vtkm/cont/DataSet.h>
#include <vtkm/filter/FieldSelection.h>
#include <vtkm/filter/mesh_info/MeshQuality.h>

namespace vtkh
{

class vtkmMeshQuality
{
public:
  vtkm::cont::DataSet Run(vtkm::cont::DataSet &input,
                          vtkm::filter::mesh_info::CellMetric metric,
                          vtkm::filter::FieldSelection map_fields);
};
}
#endif
