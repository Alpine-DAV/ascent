//-----------------------------------------------------------------------------
///
/// file: vtkm_dataset_info.cpp
///
//-----------------------------------------------------------------------------


#include <vtkh/utils/vtkm_dataset_info.hpp>

#include <vtkm/cont/Algorithm.h>

namespace vtkh
{

bool VTKMDataSetInfo::IsStructured(const vtkm::cont::DataSet &data_set,
                                   int &topo_dims)
{
  const vtkm::cont::UnknownCellSet cell_set = data_set.GetCellSet();
  return IsStructured(cell_set, topo_dims);
}

bool
VTKMDataSetInfo::IsStructured(const vtkm::rendering::Actor &actor, int &topo_dims)
{
  return IsStructured(actor.GetCells(), topo_dims);
}

bool
VTKMDataSetInfo::IsStructured(const vtkm::cont::UnknownCellSet &cell_set, int &topo_dims)
{
  bool is_structured = false;
  topo_dims = -1;

  if(cell_set.IsType<vtkm::cont::CellSetStructured<1>>())
  {
    is_structured = true;
    topo_dims = 1;
  }
  else if(cell_set.IsType<vtkm::cont::CellSetStructured<2>>())
  {
    is_structured = true;
    topo_dims = 2;
  }
  else if(cell_set.IsType<vtkm::cont::CellSetStructured<3>>())
  {
    is_structured = true;
    topo_dims = 3;
  }

  return is_structured;
}

bool
VTKMDataSetInfo::IsRectilinear(const vtkm::cont::DataSet &data_set)
{
  const vtkm::cont::CoordinateSystem coords = data_set.GetCoordinateSystem();
  return IsRectilinear(coords);
}

bool
VTKMDataSetInfo::IsRectilinear(const vtkm::rendering::Actor &actor)
{
  return IsRectilinear(actor.GetCoordinates());
}

bool
VTKMDataSetInfo::IsRectilinear(const vtkm::cont::CoordinateSystem &coords)
{

  bool is_rect= false;

  if(coords.GetData().IsType<CartesianArrayHandle>())
  {
    is_rect = true;
  }
  return is_rect;
}

bool
VTKMDataSetInfo:: IsUniform(const vtkm::cont::DataSet &data_set)
{
  const vtkm::cont::CoordinateSystem coords = data_set.GetCoordinateSystem();
  return IsUniform(coords);
}

bool
VTKMDataSetInfo::IsUniform(const vtkm::rendering::Actor &actor)
{
  return IsUniform(actor.GetCoordinates());
}

bool
VTKMDataSetInfo::IsUniform(const vtkm::cont::CoordinateSystem &coords)
{
  bool is_uniform= false;
  if(coords.GetData().IsType<UniformArrayHandle>())
  {
    is_uniform = true;
  }
  return is_uniform;
}

bool
VTKMDataSetInfo::GetPointDims(const vtkm::cont::DataSet &data_set, int *dims)
{
  const vtkm::cont::UnknownCellSet cell_set = data_set.GetCellSet();
  return GetPointDims(cell_set, dims);
}

bool
VTKMDataSetInfo::GetPointDims(const vtkm::rendering::Actor &actor, int *dims)
{
  return GetPointDims(actor.GetCells(), dims);
}

bool
VTKMDataSetInfo::GetPointDims(const vtkm::cont::UnknownCellSet &cell_set, int *dims)
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
    vtkm::cont::CellSetStructured<1> cell_set1 =
        cell_set.AsCellSet<vtkm::cont::CellSetStructured<1>>();
    vtkm::Id dims1 = cell_set1.GetPointDimensions();
    dims[0] = dims1;
  }
  else if(topo_dims == 2)
  {
    vtkm::cont::CellSetStructured<2> cell_set2 =
        cell_set.AsCellSet<vtkm::cont::CellSetStructured<2>>();
    vtkm::Id2 dims2 = cell_set2.GetPointDimensions();
    dims[0] = dims2[0];
    dims[1] = dims2[1];
  }
  else if(topo_dims == 3)
  {
    vtkm::cont::CellSetStructured<3> cell_set3 =
        cell_set.AsCellSet<vtkm::cont::CellSetStructured<3>>();
    vtkm::Id3 dims3 = cell_set3.GetPointDimensions();
    dims[0] = dims3[0];
    dims[1] = dims3[1];
    dims[2] = dims3[2];
  }

  return success;

}

bool
VTKMDataSetInfo::GetCellDims(const vtkm::cont::DataSet &data_set, int *dims)
{
  const vtkm::cont::UnknownCellSet cell_set = data_set.GetCellSet();
  return GetCellDims(cell_set, dims);
}

bool
VTKMDataSetInfo::GetCellDims(const vtkm::rendering::Actor &actor, int *dims)
{
  return GetCellDims(actor.GetCells(), dims);
}

bool
VTKMDataSetInfo::GetCellDims(const vtkm::cont::UnknownCellSet &cell_set, int *dims)
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
    vtkm::cont::CellSetStructured<1> cell_set1 =
        cell_set.AsCellSet<vtkm::cont::CellSetStructured<1>>();
    vtkm::Id dims1 = cell_set1.GetCellDimensions();
    dims[0] = dims1;
  }
  else if(topo_dims == 2)
  {
    vtkm::cont::CellSetStructured<2> cell_set2 =
        cell_set.AsCellSet<vtkm::cont::CellSetStructured<2>>();
    vtkm::Id2 dims2 = cell_set2.GetCellDimensions();
    dims[0] = dims2[0];
    dims[1] = dims2[1];
  }
  else if(topo_dims == 3)
  {
    vtkm::cont::CellSetStructured<3> cell_set3 =
        cell_set.AsCellSet<vtkm::cont::CellSetStructured<3>>();
    vtkm::Id3 dims3 = cell_set3.GetCellDimensions();
    dims[0] = dims3[0];
    dims[1] = dims3[1];
    dims[2] = dims3[2];
  }

  return success;

}

bool
VTKMDataSetInfo::IsSingleCellShape(const vtkm::cont::UnknownCellSet &cell_set, vtkm::UInt8 &shape_id)
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
    if(cell_set.IsType<vtkm::cont::CellSetSingleType<>>())
    {
      vtkm::cont::CellSetSingleType<> single = cell_set.AsCellSet<vtkm::cont::CellSetSingleType<>>();
      is_single_shape = true;
      shape_id = single.GetCellShape(0);
    }
    else if(cell_set.IsType<vtkm::cont::CellSetExplicit<>>())
    {
      vtkm::cont::CellSetExplicit<> exp = cell_set.AsCellSet<vtkm::cont::CellSetExplicit<>>();
      const vtkm::cont::ArrayHandle<vtkm::UInt8> shapes = exp.GetShapesArray(
        vtkm::TopologyElementTagCell(),
        vtkm::TopologyElementTagPoint());

      vtkm::UInt8 init_min = 255;
      vtkm::UInt8 min = vtkm::cont::Algorithm::Reduce(shapes, init_min, vtkm::Minimum());

      vtkm::UInt8 init_max = 0;
      vtkm::UInt8 max = vtkm::cont::Algorithm::Reduce(shapes, init_max, vtkm::Maximum());
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

