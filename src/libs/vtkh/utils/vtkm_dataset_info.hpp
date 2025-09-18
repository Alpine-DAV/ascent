//-----------------------------------------------------------------------------
///
/// file: viskores_dataset_info.hpp
///
//-----------------------------------------------------------------------------

#ifndef VISKORES_DATASET_INFO_HPP
#define VISKORES_DATASET_INFO_HPP

#include <vtkh/vtkh_exports.h>
#include <viskores/cont/DataSet.h>
#include <viskores/rendering/Actor.h>

namespace vtkh {

class VTKH_API VISKORESDataSetInfo
{
public:
 typedef typename viskores::cont::ArrayHandleUniformPointCoordinates UniformArrayHandle;
 typedef viskores::cont::ArrayHandle<viskores::FloatDefault> DefaultHandle;
 typedef typename viskores::cont::ArrayHandleCartesianProduct<DefaultHandle,
                                                          DefaultHandle,
                                                          DefaultHandle> CartesianArrayHandle;
//-----------------------------------------------------------------------------
  static bool IsStructured(const viskores::cont::DataSet &data_set,
                           int &topo_dims);

  static bool IsStructured(const viskores::rendering::Actor &actor, int &topo_dims);

  static bool IsStructured(const viskores::cont::UnknownCellSet &cell_set, int &topo_dims);

  static bool IsRectilinear(const viskores::cont::DataSet &data_set);

  static bool IsRectilinear(const viskores::rendering::Actor &actor);

  static bool IsRectilinear(const viskores::cont::CoordinateSystem &coords);

  static bool IsUniform(const viskores::cont::DataSet &data_set);

  static bool IsUniform(const viskores::rendering::Actor &actor);

  static bool IsUniform(const viskores::cont::CoordinateSystem &coords);

  static bool GetPointDims(const viskores::cont::DataSet &data_set, int *dims);

  static bool GetPointDims(const viskores::rendering::Actor &actor, int *dims);

  static bool GetPointDims(const viskores::cont::UnknownCellSet &cell_set, int *dims);

  static bool GetCellDims(const viskores::cont::DataSet &data_set, int *dims);

  static bool GetCellDims(const viskores::rendering::Actor &actor, int *dims);

  static bool GetCellDims(const viskores::cont::UnknownCellSet &cell_set, int *dims);

  static bool IsSingleCellShape(const viskores::cont::UnknownCellSet &cell_set, viskores::UInt8 &shape_id);

};

} // namespace vtkh

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
