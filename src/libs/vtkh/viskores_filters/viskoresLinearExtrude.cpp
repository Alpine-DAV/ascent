#include "viskoresLinearExtrude.hpp"

#include <vtkh/Error.hpp>

#include <viskores/TypeList.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleSOA.h>
#include <viskores/cont/ArrayHandleSOAStride.h>
#include <viskores/cont/CastAndCall.h>
#include <viskores/cont/CellSetExplicit.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/CoordinateSystem.h>
#include <viskores/cont/StorageList.h>
#include <viskores/cont/UnknownCellSet.h>

#include <string>
#include <vector>

namespace vtkh
{

namespace detail
{

// Cast 2D/3D coordinate arrays to a host-side Vec<Float64,3> vector for downstream processing.
struct CoordsToVec3d
{
  std::vector<viskores::Vec<viskores::Float64,3>> &Coords;

  explicit CoordsToVec3d(std::vector<viskores::Vec<viskores::Float64,3>> &coords)
    : Coords(coords)
  {}

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<viskores::Vec<T,3>,S> &in) const
  {
    const viskores::Id num_values = in.GetNumberOfValues();
    this->Coords.resize(static_cast<size_t>(num_values));
    auto portal = in.ReadPortal();
    for(viskores::Id i = 0; i < num_values; ++i)
    {
      const auto coord_value = portal.Get(i);
      this->Coords[static_cast<size_t>(i)] =
        viskores::Vec<viskores::Float64,3>(static_cast<viskores::Float64>(coord_value[0]),
                                           static_cast<viskores::Float64>(coord_value[1]),
                                           static_cast<viskores::Float64>(coord_value[2]));
    }
  }

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<viskores::Vec<T,2>,S> &in) const
  {
    const viskores::Id num_values = in.GetNumberOfValues();
    this->Coords.resize(static_cast<size_t>(num_values));
    auto portal = in.ReadPortal();
    for(viskores::Id i = 0; i < num_values; ++i)
    {
      const auto coord_value = portal.Get(i);
      this->Coords[static_cast<size_t>(i)] = viskores::Vec<viskores::Float64,3>(static_cast<viskores::Float64>(coord_value[0]),
                                                                                static_cast<viskores::Float64>(coord_value[1]),
                                                                                0.0);
    }
  }

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<T,S> &) const
  {
    throw Error("vtkh::LinearExtrude expects a 2D or 3D coordinate system");
  }
};

// Replicate a point- or cell-associated field across the generated planes/steps.
struct ReplicateField
{
  viskores::cont::DataSet &Out;
  std::string Name;
  viskores::cont::Field::Association Assoc;
  viskores::Id BaseSize;
  viskores::Id Replications;

  ReplicateField(viskores::cont::DataSet &out,
                 const std::string &name,
                 const viskores::cont::Field::Association assoc,
                 const viskores::Id base_size,
                 const viskores::Id replications)
    : Out(out),
      Name(name),
      Assoc(assoc),
      BaseSize(base_size),
      Replications(replications)
  {}

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<T,S> &in) const
  {
    viskores::cont::ArrayHandle<T> out;
    out.Allocate(this->BaseSize * this->Replications);

    for(viskores::Id replication_index = 0; replication_index < this->Replications; ++replication_index)
    {
      const viskores::Id out_offset = replication_index * this->BaseSize;
      viskores::cont::Algorithm::CopySubRange(in, 0, this->BaseSize, out, out_offset);
    }

    this->Out.AddField(viskores::cont::Field(this->Name, this->Assoc, out));
  }
};

} // namespace detail

// Linearly extrude an input mesh along a vector for a fixed number of steps, then replicate selected fields.
viskores::cont::DataSet
viskoresLinearExtrude::Run(viskores::cont::DataSet &input,
                           const viskores::Vec<viskores::Float64,3> &vector,
                           const viskores::Int32 steps,
                           viskores::filter::FieldSelection map_fields)
{
  if(steps <= 0)
  {
    throw Error("vtkh::LinearExtrude requires 'steps' > 0");
  }

  // Extract connectivity from the supported unstructured cell set types.
  const viskores::cont::UnknownCellSet unknown_cs = input.GetCellSet();
  viskores::cont::ArrayHandle<viskores::Id> conn_ids;
  viskores::Id num_cells = 0;

  if(unknown_cs.IsType<viskores::cont::CellSetSingleType<>>())
  {
    const auto cs = unknown_cs.AsCellSet<viskores::cont::CellSetSingleType<>>();
    num_cells = cs.GetNumberOfCells();
    conn_ids = cs.GetConnectivityArray(viskores::TopologyElementTagCell(),
                                       viskores::TopologyElementTagPoint());
  }
  else if(unknown_cs.IsType<viskores::cont::CellSetExplicit<>>())
  {
    const auto cs = unknown_cs.AsCellSet<viskores::cont::CellSetExplicit<>>();
    num_cells = cs.GetNumberOfCells();
    conn_ids = cs.GetConnectivityArray(viskores::TopologyElementTagCell(),
                                       viskores::TopologyElementTagPoint());
  }
  else
  {
    throw Error("vtkh::LinearExtrude expects an unstructured cell set");
  }

  // Validate mesh sizes and derive point/cell counts used for extrusion.
  const viskores::Id num_points_per_plane = input.GetCoordinateSystem(0).GetData().GetNumberOfValues();
  if(num_points_per_plane <= 0 || num_cells <= 0)
  {
    return viskores::cont::DataSet{};
  }

  const viskores::Id conn_size = conn_ids.GetNumberOfValues();
  if(conn_size <= 0)
  {
    throw Error("vtkh::LinearExtrude requires non-empty connectivity");
  }

  if(conn_size % num_cells != 0)
  {
    throw Error("vtkh::LinearExtrude only supports fixed-size unstructured cells");
  }

  const viskores::Id points_per_cell = conn_size / num_cells;
  if(points_per_cell != 2 && points_per_cell != 3 && points_per_cell != 4)
  {
    throw Error("vtkh::LinearExtrude only supports line (2), triangle (3), or quad (4) meshes");
  }

  std::vector<viskores::Int32> conn32_vec(static_cast<size_t>(conn_size));
  auto conn_portal = conn_ids.ReadPortal();
  for(viskores::Id i = 0; i < conn_size; ++i)
  {
    conn32_vec[static_cast<size_t>(i)] = static_cast<viskores::Int32>(conn_portal.Get(i));
  }

  const viskores::Id planes = static_cast<viskores::Id>(steps + 1);
  const viskores::Id num_out_points = num_points_per_plane * planes;

  // Read the input coordinates into a host vector for repeated translation.
  std::vector<viskores::Vec<viskores::Float64,3>> base_coords;
  {
    auto coords = input.GetCoordinateSystem(0).GetData();
    detail::CoordsToVec3d to_vec(base_coords);
    using CoordValueTypes = viskores::List<viskores::Vec<viskores::Float32,2>,
                                           viskores::Vec<viskores::Float64,2>,
                                           viskores::Vec<viskores::Float32,3>,
                                           viskores::Vec<viskores::Float64,3>>;
    using CoordStorageTypes = viskores::List<viskores::cont::StorageTagBasic,
                                             viskores::cont::StorageTagSOA,
                                             viskores::cont::StorageTagSOAStride>;
    coords.CastAndCallForTypes<CoordValueTypes, CoordStorageTypes>(to_vec);
  }

  viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64,3>> out_coords;
  out_coords.Allocate(num_out_points);

  // Generate translated coordinates for each extrusion plane.
  {
    auto out_portal = out_coords.WritePortal();
    for(viskores::Int32 p = 0; p < planes; ++p)
    {
      const viskores::Float64 t = static_cast<viskores::Float64>(p) / static_cast<viskores::Float64>(steps);
      const viskores::Vec<viskores::Float64,3> delta = vector * t;
      const viskores::Id point_offset = static_cast<viskores::Id>(p) * num_points_per_plane;
      for(viskores::Id i = 0; i < num_points_per_plane; ++i)
      {
        out_portal.Set(point_offset + i, base_coords[static_cast<size_t>(i)] + delta);
      }
    }
  }

  viskores::cont::CellSetSingleType<> out_cs;
  const viskores::Id out_cells = num_cells * static_cast<viskores::Id>(steps);

  if(points_per_cell == 2)
  {
    // Lines extruded across steps become quad cells.
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 4);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 step_index = 0; step_index < steps; ++step_index)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(step_index);
      const viskores::Id plane1 = static_cast<viskores::Id>(step_index + 1);
      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id cell_index = 0; cell_index < num_cells; ++cell_index)
      {
        const viskores::Id base_point_id0 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 2 + 0)]);
        const viskores::Id base_point_id1 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 2 + 1)]);

        const viskores::Id out_cell_id = static_cast<viskores::Id>(step_index) * num_cells + cell_index;
        const viskores::Id out_off = out_cell_id * 4;
        out_conn_portal.Set(out_off + 0, p0_off + base_point_id0);
        out_conn_portal.Set(out_off + 1, p0_off + base_point_id1);
        out_conn_portal.Set(out_off + 2, p1_off + base_point_id1);
        out_conn_portal.Set(out_off + 3, p1_off + base_point_id0);
      }
    }

    out_cs.Fill(num_out_points, viskores::CELL_SHAPE_QUAD, 4, out_conn);
  }
  else if(points_per_cell == 3)
  {
    // Triangles extruded across steps become wedge cells (prisms).
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 6);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 step_index = 0; step_index < steps; ++step_index)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(step_index);
      const viskores::Id plane1 = static_cast<viskores::Id>(step_index + 1);
      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id cell_index = 0; cell_index < num_cells; ++cell_index)
      {
        const viskores::Id base_point_id0 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 3 + 0)]);
        const viskores::Id base_point_id1 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 3 + 1)]);
        const viskores::Id base_point_id2 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 3 + 2)]);

        const viskores::Id out_cell_id = static_cast<viskores::Id>(step_index) * num_cells + cell_index;
        const viskores::Id out_off = out_cell_id * 6;
        out_conn_portal.Set(out_off + 0, p0_off + base_point_id0);
        out_conn_portal.Set(out_off + 1, p0_off + base_point_id1);
        out_conn_portal.Set(out_off + 2, p0_off + base_point_id2);
        out_conn_portal.Set(out_off + 3, p1_off + base_point_id0);
        out_conn_portal.Set(out_off + 4, p1_off + base_point_id1);
        out_conn_portal.Set(out_off + 5, p1_off + base_point_id2);
      }
    }

    out_cs.Fill(num_out_points, viskores::CELL_SHAPE_WEDGE, 6, out_conn);
  }
  else
  {
    // Quads extruded across steps become hex cells.
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 8);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 step_index = 0; step_index < steps; ++step_index)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(step_index);
      const viskores::Id plane1 = static_cast<viskores::Id>(step_index + 1);
      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id cell_index = 0; cell_index < num_cells; ++cell_index)
      {
        const viskores::Id base_point_id0 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 4 + 0)]);
        const viskores::Id base_point_id1 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 4 + 1)]);
        const viskores::Id base_point_id2 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 4 + 2)]);
        const viskores::Id base_point_id3 = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(cell_index * 4 + 3)]);

        const viskores::Id out_cell_id = static_cast<viskores::Id>(step_index) * num_cells + cell_index;
        const viskores::Id out_off = out_cell_id * 8;
        out_conn_portal.Set(out_off + 0, p0_off + base_point_id0);
        out_conn_portal.Set(out_off + 1, p0_off + base_point_id1);
        out_conn_portal.Set(out_off + 2, p0_off + base_point_id2);
        out_conn_portal.Set(out_off + 3, p0_off + base_point_id3);
        out_conn_portal.Set(out_off + 4, p1_off + base_point_id0);
        out_conn_portal.Set(out_off + 5, p1_off + base_point_id1);
        out_conn_portal.Set(out_off + 6, p1_off + base_point_id2);
        out_conn_portal.Set(out_off + 7, p1_off + base_point_id3);
      }
    }

    out_cs.Fill(num_out_points, viskores::CELL_SHAPE_HEXAHEDRON, 8, out_conn);
  }

  // Assemble output dataset (cell set + coordinates) and replicate requested fields.
  viskores::cont::DataSet output;
  output.SetCellSet(out_cs);
  output.AddCoordinateSystem(
    viskores::cont::CoordinateSystem(input.GetCoordinateSystem(0).GetName(), out_coords));

  const viskores::Id rep_cells = static_cast<viskores::Id>(steps);
  using FieldStorageTypes = viskores::List<viskores::cont::StorageTagBasic,
                                           viskores::cont::StorageTagSOA,
                                           viskores::cont::StorageTagSOAStride,
                                           viskores::cont::StorageTagUniformPoints,
                                          viskores::cont::StorageTagCartesianProduct<viskores::cont::StorageTagBasic,
                                                                                     viskores::cont::StorageTagBasic,
                                                                                     viskores::cont::StorageTagBasic>>;

  for(viskores::IdComponent i = 0; i < input.GetNumberOfFields(); ++i)
  {
    const viskores::cont::Field &f = input.GetField(i);
    // Coordinate systems may also appear as fields; do not replicate them as ordinary fields.
    if(input.HasCoordinateSystem(f.GetName()))
    {
      continue;
    }
    if(!map_fields.IsFieldSelected(f))
    {
      continue;
    }

    if(f.GetAssociation() == viskores::cont::Field::Association::Points)
    {
      detail::ReplicateField copier(output,
                                    f.GetName(),
                                    f.GetAssociation(),
                                    num_points_per_plane,
                                    planes);
      f.GetData().CastAndCallForTypes<viskores::TypeListAll, FieldStorageTypes>(copier);
    }
    else if(f.GetAssociation() == viskores::cont::Field::Association::Cells)
    {
      detail::ReplicateField copier(output,
                                    f.GetName(),
                                    f.GetAssociation(),
                                    num_cells,
                                    rep_cells);
      f.GetData().CastAndCallForTypes<viskores::TypeListAll, FieldStorageTypes>(copier);
    }
  }

  return output;
}

} // namespace vtkh
