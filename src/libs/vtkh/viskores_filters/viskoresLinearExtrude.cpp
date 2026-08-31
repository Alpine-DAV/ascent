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

struct CoordsToVec3d
{
  std::vector<viskores::Vec<viskores::Float64,3>> &Coords;

  explicit CoordsToVec3d(std::vector<viskores::Vec<viskores::Float64,3>> &coords)
    : Coords(coords)
  {}

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<viskores::Vec<T,3>,S> &in) const
  {
    const viskores::Id n = in.GetNumberOfValues();
    this->Coords.resize(static_cast<size_t>(n));
    auto portal = in.ReadPortal();
    for(viskores::Id i = 0; i < n; ++i)
    {
      const auto v = portal.Get(i);
      this->Coords[static_cast<size_t>(i)] =
        viskores::Vec<viskores::Float64,3>(static_cast<viskores::Float64>(v[0]),
                                           static_cast<viskores::Float64>(v[1]),
                                           static_cast<viskores::Float64>(v[2]));
    }
  }

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<viskores::Vec<T,2>,S> &in) const
  {
    const viskores::Id n = in.GetNumberOfValues();
    this->Coords.resize(static_cast<size_t>(n));
    auto portal = in.ReadPortal();
    for(viskores::Id i = 0; i < n; ++i)
    {
      const auto v = portal.Get(i);
      this->Coords[static_cast<size_t>(i)] = viskores::Vec<viskores::Float64,3>(static_cast<viskores::Float64>(v[0]),
                                                                                static_cast<viskores::Float64>(v[1]),
                                                                                0.0);
    }
  }

  template<typename T, typename S>
  void operator()(const viskores::cont::ArrayHandle<T,S> &) const
  {
    throw Error("vtkh::LinearExtrude expects a 2D or 3D coordinate system");
  }
};

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

    for(viskores::Id r = 0; r < this->Replications; ++r)
    {
      const viskores::Id out_offset = r * this->BaseSize;
      viskores::cont::Algorithm::CopySubRange(in, 0, this->BaseSize, out, out_offset);
    }

    this->Out.AddField(viskores::cont::Field(this->Name, this->Assoc, out));
  }
};

} // namespace detail

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
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 4);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 s = 0; s < steps; ++s)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(s);
      const viskores::Id plane1 = static_cast<viskores::Id>(s + 1);
      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id c = 0; c < num_cells; ++c)
      {
        const viskores::Id a = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 2 + 0)]);
        const viskores::Id b = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 2 + 1)]);

        const viskores::Id out_cell_id = static_cast<viskores::Id>(s) * num_cells + c;
        const viskores::Id out_off = out_cell_id * 4;
        out_conn_portal.Set(out_off + 0, p0_off + a);
        out_conn_portal.Set(out_off + 1, p0_off + b);
        out_conn_portal.Set(out_off + 2, p1_off + b);
        out_conn_portal.Set(out_off + 3, p1_off + a);
      }
    }

    out_cs.Fill(num_out_points, viskores::CELL_SHAPE_QUAD, 4, out_conn);
  }
  else if(points_per_cell == 3)
  {
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 6);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 s = 0; s < steps; ++s)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(s);
      const viskores::Id plane1 = static_cast<viskores::Id>(s + 1);
      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id c = 0; c < num_cells; ++c)
      {
        const viskores::Id a = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 3 + 0)]);
        const viskores::Id b = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 3 + 1)]);
        const viskores::Id cc = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 3 + 2)]);

        const viskores::Id out_cell_id = static_cast<viskores::Id>(s) * num_cells + c;
        const viskores::Id out_off = out_cell_id * 6;
        out_conn_portal.Set(out_off + 0, p0_off + a);
        out_conn_portal.Set(out_off + 1, p0_off + b);
        out_conn_portal.Set(out_off + 2, p0_off + cc);
        out_conn_portal.Set(out_off + 3, p1_off + a);
        out_conn_portal.Set(out_off + 4, p1_off + b);
        out_conn_portal.Set(out_off + 5, p1_off + cc);
      }
    }

    out_cs.Fill(num_out_points, viskores::CELL_SHAPE_WEDGE, 6, out_conn);
  }
  else
  {
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 8);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 s = 0; s < steps; ++s)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(s);
      const viskores::Id plane1 = static_cast<viskores::Id>(s + 1);
      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id c = 0; c < num_cells; ++c)
      {
        const viskores::Id a = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 4 + 0)]);
        const viskores::Id b = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 4 + 1)]);
        const viskores::Id cc = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 4 + 2)]);
        const viskores::Id d = static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 4 + 3)]);

        const viskores::Id out_cell_id = static_cast<viskores::Id>(s) * num_cells + c;
        const viskores::Id out_off = out_cell_id * 8;
        out_conn_portal.Set(out_off + 0, p0_off + a);
        out_conn_portal.Set(out_off + 1, p0_off + b);
        out_conn_portal.Set(out_off + 2, p0_off + cc);
        out_conn_portal.Set(out_off + 3, p0_off + d);
        out_conn_portal.Set(out_off + 4, p1_off + a);
        out_conn_portal.Set(out_off + 5, p1_off + b);
        out_conn_portal.Set(out_off + 6, p1_off + cc);
        out_conn_portal.Set(out_off + 7, p1_off + d);
      }
    }

    out_cs.Fill(num_out_points, viskores::CELL_SHAPE_HEXAHEDRON, 8, out_conn);
  }

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
