#include "viskoresRevolve.hpp"

#include <vtkh/Error.hpp>

#include <viskores/Math.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/CastAndCall.h>
#include <viskores/cont/CellSetExplicit.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/CoordinateSystem.h>
#include <viskores/cont/UnknownArrayHandle.h>
#include <viskores/cont/UnknownCellSet.h>

#include <cmath>
#include <string>
#include <vector>

namespace vtkh
{

namespace detail
{

VISKORES_EXEC_CONT
inline viskores::Vec<viskores::Float64,3>
rotate_about_axis(const viskores::Vec<viskores::Float64,3> &p,
                  const viskores::Vec<viskores::Float64,3> &origin,
                  const viskores::Vec<viskores::Float64,3> &axis_unit,
                  const viskores::Float64 angle_radians)
{
  const viskores::Vec<viskores::Float64,3> v = p - origin;
  const viskores::Float64 c = viskores::Cos(angle_radians);
  const viskores::Float64 s = viskores::Sin(angle_radians);

  // Rodrigues' rotation formula (implemented explicitly to avoid any ambiguity
  // around vector-analysis helpers across backends).
  const viskores::Float64 ux = axis_unit[0];
  const viskores::Float64 uy = axis_unit[1];
  const viskores::Float64 uz = axis_unit[2];

  const viskores::Float64 vx = v[0];
  const viskores::Float64 vy = v[1];
  const viskores::Float64 vz = v[2];

  const viskores::Float64 dot = ux * vx + uy * vy + uz * vz;

  const viskores::Float64 cx = uy * vz - uz * vy;
  const viskores::Float64 cy = uz * vx - ux * vz;
  const viskores::Float64 cz = ux * vy - uy * vx;

  const viskores::Float64 one_minus_c = 1.0 - c;

  const viskores::Vec<viskores::Float64,3> term1(vx * c, vy * c, vz * c);
  const viskores::Vec<viskores::Float64,3> term2(cx * s, cy * s, cz * s);
  const viskores::Vec<viskores::Float64,3> term3(ux * dot * one_minus_c,
                                                 uy * dot * one_minus_c,
                                                 uz * dot * one_minus_c);

  return origin + term1 + term2 + term3;
}

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
  void operator()(const viskores::cont::ArrayHandle<T,S> &) const
  {
    throw Error("vtkh::Revolve expects a 3D coordinate system");
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
viskoresRevolve::Run(viskores::cont::DataSet &input,
                     const viskores::Vec<viskores::Float64,3> &point,
                     const viskores::Vec<viskores::Float64,3> &axis,
                     const viskores::Float64 start_angle_degrees,
                     const viskores::Float64 sweep_angle_degrees,
                     const viskores::Int32 steps,
                     const bool periodic,
                     viskores::filter::FieldSelection map_fields)
{
  if(steps <= 0)
  {
    throw Error("vtkh::Revolve requires 'steps' > 0");
  }

  const viskores::cont::UnknownCellSet unknown_cs = input.GetCellSet();
  viskores::cont::ArrayHandle<viskores::Id> conn_ids;
  viskores::Id num_cells = 0;

  if(unknown_cs.IsType<viskores::cont::CellSetSingleType<>>())
  {
    const viskores::cont::CellSetSingleType<> cs =
      unknown_cs.AsCellSet<viskores::cont::CellSetSingleType<>>();
    num_cells = cs.GetNumberOfCells();
    conn_ids = cs.GetConnectivityArray(viskores::TopologyElementTagCell(),
                                       viskores::TopologyElementTagPoint());
  }
  else if(unknown_cs.IsType<viskores::cont::CellSetExplicit<>>())
  {
    const viskores::cont::CellSetExplicit<> cs =
      unknown_cs.AsCellSet<viskores::cont::CellSetExplicit<>>();
    num_cells = cs.GetNumberOfCells();
    conn_ids = cs.GetConnectivityArray(viskores::TopologyElementTagCell(),
                                       viskores::TopologyElementTagPoint());
  }
  else
  {
    throw Error("vtkh::Revolve expects a triangle-only unstructured cell set");
  }

  const viskores::Id num_points_per_plane = input.GetCoordinateSystem(0).GetData().GetNumberOfValues();
  if(num_points_per_plane <= 0 || num_cells <= 0)
  {
    return viskores::cont::DataSet{};
  }

  const viskores::Id conn_size = conn_ids.GetNumberOfValues();
  if(conn_size <= 0)
  {
    throw Error("vtkh::Revolve requires non-empty connectivity");
  }

  if(conn_size % num_cells != 0)
  {
    throw Error("vtkh::Revolve only supports fixed-size unstructured cells");
  }

  const viskores::Id points_per_cell = conn_size / num_cells;
  if(points_per_cell != 2 && points_per_cell != 3)
  {
    throw Error("vtkh::Revolve only supports line (2) or triangle (3) meshes");
  }

  std::vector<viskores::Int32> conn32_vec(static_cast<size_t>(conn_size));
  auto conn_portal = conn_ids.ReadPortal();
  for(viskores::Id i = 0; i < conn_size; ++i)
  {
    conn32_vec[static_cast<size_t>(i)] = static_cast<viskores::Int32>(conn_portal.Get(i));
  }

  const viskores::Int32 planes = periodic ? steps : (steps + 1);
  const viskores::Id num_out_points =
    num_points_per_plane * static_cast<viskores::Id>(planes);

  std::vector<viskores::Vec<viskores::Float64,3>> base_coords;
  {
    auto coords = input.GetCoordinateSystem(0).GetData();
    detail::CoordsToVec3d to_vec(base_coords);
    coords.CastAndCall(to_vec);
  }

  viskores::Vec<viskores::Float64,3> axis_unit = axis;
  const viskores::Float64 axis_mag = viskores::Magnitude(axis_unit);
  if(axis_mag <= 0.0)
  {
    throw Error("vtkh::Revolve requires a non-zero 'axis'");
  }
  axis_unit = axis_unit / axis_mag;

  const viskores::Float64 start_radians = viskores::Pi() * start_angle_degrees / 180.0;
  const viskores::Float64 sweep_radians = viskores::Pi() * sweep_angle_degrees / 180.0;
  const viskores::Float64 delta = sweep_radians / static_cast<viskores::Float64>(steps);

  viskores::cont::ArrayHandle<viskores::Vec<viskores::Float64,3>> out_coords;
  out_coords.Allocate(num_out_points);

  {
    auto out_portal = out_coords.WritePortal();
    for(viskores::Int32 p = 0; p < planes; ++p)
    {
      const viskores::Float64 theta = start_radians + static_cast<viskores::Float64>(p) * delta;
      const viskores::Id point_offset =
        static_cast<viskores::Id>(p) * num_points_per_plane;
      for(viskores::Id i = 0; i < num_points_per_plane; ++i)
      {
        const auto rp = detail::rotate_about_axis(base_coords[static_cast<size_t>(i)],
                                                  point,
                                                  axis_unit,
                                                  theta);
        out_portal.Set(point_offset + i, rp);
      }
    }
  }

  viskores::cont::CellSetSingleType<> out_cs;
  if(points_per_cell == 3)
  {
    const viskores::Id out_cells = num_cells * static_cast<viskores::Id>(steps);
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 6);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 s = 0; s < steps; ++s)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(s);
      const viskores::Id plane1 = periodic
        ? static_cast<viskores::Id>((s + 1) % steps)
        : static_cast<viskores::Id>(s + 1);

      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id c = 0; c < num_cells; ++c)
      {
        const viskores::Id a =
          static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 3 + 0)]);
        const viskores::Id b =
          static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 3 + 1)]);
        const viskores::Id cc =
          static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 3 + 2)]);

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
    const viskores::Id out_cells = num_cells * static_cast<viskores::Id>(steps);
    viskores::cont::ArrayHandle<viskores::Id> out_conn;
    out_conn.Allocate(out_cells * 4);
    auto out_conn_portal = out_conn.WritePortal();

    for(viskores::Int32 s = 0; s < steps; ++s)
    {
      const viskores::Id plane0 = static_cast<viskores::Id>(s);
      const viskores::Id plane1 = periodic
        ? static_cast<viskores::Id>((s + 1) % steps)
        : static_cast<viskores::Id>(s + 1);

      const viskores::Id p0_off = plane0 * num_points_per_plane;
      const viskores::Id p1_off = plane1 * num_points_per_plane;

      for(viskores::Id c = 0; c < num_cells; ++c)
      {
        const viskores::Id a =
          static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 2 + 0)]);
        const viskores::Id b =
          static_cast<viskores::Id>(conn32_vec[static_cast<size_t>(c * 2 + 1)]);

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

  viskores::cont::DataSet output;
  output.SetCellSet(out_cs);
  output.AddCoordinateSystem(viskores::cont::CoordinateSystem(input.GetCoordinateSystem(0).GetName(),
                                                              out_coords));

  const viskores::Id rep_cells = static_cast<viskores::Id>(steps);

  for(viskores::IdComponent i = 0; i < input.GetNumberOfFields(); ++i)
  {
    const viskores::cont::Field &f = input.GetField(i);
    // Coordinate systems may also appear as fields; do not replicate them as
    // ordinary fields since they can overwrite the output coordinate system.
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
                                    static_cast<viskores::Id>(planes));
      viskores::cont::CastAndCall(f.GetData(), copier);
    }
    else if(f.GetAssociation() == viskores::cont::Field::Association::Cells)
    {
      detail::ReplicateField copier(output,
                                    f.GetName(),
                                    f.GetAssociation(),
                                    num_cells,
                                    rep_cells);
      viskores::cont::CastAndCall(f.GetData(), copier);
    }
  }

  return output;
}

} // namespace vtkh
