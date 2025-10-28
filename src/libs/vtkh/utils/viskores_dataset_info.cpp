//-----------------------------------------------------------------------------
///
/// file: viskores_dataset_info.cpp
///
//-----------------------------------------------------------------------------


#include <vtkh/utils/viskores_dataset_info.hpp>

#include <viskores/cont/Algorithm.h>

namespace vtkh
{

bool VISKORESDataSetInfo::IsStructured(const viskores::cont::DataSet &data_set,
                                   int &topo_dims)
{
  const viskores::cont::UnknownCellSet cell_set = data_set.GetCellSet();
  return IsStructured(cell_set, topo_dims);
}

bool
VISKORESDataSetInfo::IsStructured(const viskores::rendering::Actor &actor, int &topo_dims)
{
  return IsStructured(actor.GetCells(), topo_dims);
}

bool
VISKORESDataSetInfo::IsStructured(const viskores::cont::UnknownCellSet &cell_set, int &topo_dims)
{
  bool is_structured = false;
  topo_dims = -1;

  if(cell_set.IsType<viskores::cont::CellSetStructured<1>>())
  {
    is_structured = true;
    topo_dims = 1;
  }
  else if(cell_set.IsType<viskores::cont::CellSetStructured<2>>())
  {
    is_structured = true;
    topo_dims = 2;
  }
  else if(cell_set.IsType<viskores::cont::CellSetStructured<3>>())
  {
    is_structured = true;
    topo_dims = 3;
  }

  return is_structured;
}

bool
VISKORESDataSetInfo::IsRectilinear(const viskores::cont::DataSet &data_set)
{
  const viskores::cont::CoordinateSystem coords = data_set.GetCoordinateSystem();
  return IsRectilinear(coords);
}

bool
VISKORESDataSetInfo::IsRectilinear(const viskores::rendering::Actor &actor)
{
  return IsRectilinear(actor.GetCoordinates());
}

bool
VISKORESDataSetInfo::IsRectilinear(const viskores::cont::CoordinateSystem &coords)
{

  bool is_rect= false;

  if(coords.GetData().IsType<CartesianArrayHandle>())
  {
    is_rect = true;
  }
  return is_rect;
}

bool
VISKORESDataSetInfo:: IsUniform(const viskores::cont::DataSet &data_set)
{
  const viskores::cont::CoordinateSystem coords = data_set.GetCoordinateSystem();
  return IsUniform(coords);
}

bool
VISKORESDataSetInfo::IsUniform(const viskores::rendering::Actor &actor)
{
  return IsUniform(actor.GetCoordinates());
}

bool
VISKORESDataSetInfo::IsUniform(const viskores::cont::CoordinateSystem &coords)
{
  bool is_uniform= false;
  if(coords.GetData().IsType<UniformArrayHandle>())
  {
    is_uniform = true;
  }
  return is_uniform;
}

bool
VISKORESDataSetInfo::GetPointDims(const viskores::cont::DataSet &data_set, int *dims)
{
  const viskores::cont::UnknownCellSet cell_set = data_set.GetCellSet();
  return GetPointDims(cell_set, dims);
}

bool
VISKORESDataSetInfo::GetPointDims(const viskores::rendering::Actor &actor, int *dims)
{
  return GetPointDims(actor.GetCells(), dims);
}

bool
VISKORESDataSetInfo::GetPointDims(const viskores::cont::UnknownCellSet &cell_set, int *dims)
{
  int topo_dims;
  bool is_structured = IsStructured(cell_set, topo_dims);
  bool success = false;
  if(!is_structured)
  {
    return success;
  }
  else
  {
    success = true;
  }

  if(topo_dims == 1)
  {
    viskores::cont::CellSetStructured<1> cell_set1 =
        cell_set.AsCellSet<viskores::cont::CellSetStructured<1>>();
    viskores::Id dims1 = cell_set1.GetPointDimensions();
    dims[0] = dims1;
  }
  else if(topo_dims == 2)
  {
    viskores::cont::CellSetStructured<2> cell_set2 =
        cell_set.AsCellSet<viskores::cont::CellSetStructured<2>>();
    viskores::Id2 dims2 = cell_set2.GetPointDimensions();
    dims[0] = dims2[0];
    dims[1] = dims2[1];
  }
  else if(topo_dims == 3)
  {
    viskores::cont::CellSetStructured<3> cell_set3 =
        cell_set.AsCellSet<viskores::cont::CellSetStructured<3>>();
    viskores::Id3 dims3 = cell_set3.GetPointDimensions();
    dims[0] = dims3[0];
    dims[1] = dims3[1];
    dims[2] = dims3[2];
  }

  return success;

}

bool
VISKORESDataSetInfo::GetCellDims(const viskores::cont::DataSet &data_set, int *dims)
{
  const viskores::cont::UnknownCellSet cell_set = data_set.GetCellSet();
  return GetCellDims(cell_set, dims);
}

bool
VISKORESDataSetInfo::GetCellDims(const viskores::rendering::Actor &actor, int *dims)
{
  return GetCellDims(actor.GetCells(), dims);
}

bool
VISKORESDataSetInfo::GetCellDims(const viskores::cont::UnknownCellSet &cell_set, int *dims)
{
  int topo_dims;
  bool is_structured = IsStructured(cell_set, topo_dims);
  bool success = false;
  if(!is_structured)
  {
    return success;
  }
  else
  {
    success = true;
  }

  if(topo_dims == 1)
  {
    viskores::cont::CellSetStructured<1> cell_set1 =
        cell_set.AsCellSet<viskores::cont::CellSetStructured<1>>();
    viskores::Id dims1 = cell_set1.GetCellDimensions();
    dims[0] = dims1;
  }
  else if(topo_dims == 2)
  {
    viskores::cont::CellSetStructured<2> cell_set2 =
        cell_set.AsCellSet<viskores::cont::CellSetStructured<2>>();
    viskores::Id2 dims2 = cell_set2.GetCellDimensions();
    dims[0] = dims2[0];
    dims[1] = dims2[1];
  }
  else if(topo_dims == 3)
  {
    viskores::cont::CellSetStructured<3> cell_set3 =
        cell_set.AsCellSet<viskores::cont::CellSetStructured<3>>();
    viskores::Id3 dims3 = cell_set3.GetCellDimensions();
    dims[0] = dims3[0];
    dims[1] = dims3[1];
    dims[2] = dims3[2];
  }

  return success;

}

bool
VISKORESDataSetInfo::IsSingleCellShape(const viskores::cont::UnknownCellSet &cell_set, viskores::UInt8 &shape_id)
{
  int dims;
  shape_id = 0;
  bool is_single_shape = false;
  if(IsStructured(cell_set, dims))
  {
    is_single_shape = true;
    shape_id = 12;
  }
  else
  {
    // we have an explicit cell set so we have to look deeper
    if(cell_set.IsType<viskores::cont::CellSetSingleType<>>())
    {
      viskores::cont::CellSetSingleType<> single = cell_set.AsCellSet<viskores::cont::CellSetSingleType<>>();
      is_single_shape = true;
      shape_id = single.GetCellShape(0);
    }
    else if(cell_set.IsType<viskores::cont::CellSetExplicit<>>())
    {
      viskores::cont::CellSetExplicit<> exp = cell_set.AsCellSet<viskores::cont::CellSetExplicit<>>();
      const viskores::cont::ArrayHandle<viskores::UInt8> shapes = exp.GetShapesArray(
        viskores::TopologyElementTagCell(),
        viskores::TopologyElementTagPoint());

      viskores::UInt8 init_min = 255;
      viskores::UInt8 min = viskores::cont::Algorithm::Reduce(shapes, init_min, viskores::Minimum());

      viskores::UInt8 init_max = 0;
      viskores::UInt8 max = viskores::cont::Algorithm::Reduce(shapes, init_max, viskores::Maximum());
      if(min == max)
      {
        is_single_shape = true;
        shape_id = max;
      }
    }

  }

  return is_single_shape;
}

} // namespace vtkh

