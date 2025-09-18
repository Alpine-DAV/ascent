#ifndef VTK_H_VISKORES_MESH_QUALITY_HPP
#define VTK_H_VISKORES_MESH_QUALITY_HPP

#include <viskores/cont/DataSet.h>
#include <viskores/filter/FieldSelection.h>
#include <viskores/filter/mesh_info/MeshQuality.h>

namespace vtkh
{

class viskoresMeshQuality
{
public:
  viskores::cont::DataSet Run(viskores::cont::DataSet &input,
                          viskores::filter::mesh_info::CellMetric metric,
                          viskores::filter::FieldSelection map_fields);
};
}
#endif
