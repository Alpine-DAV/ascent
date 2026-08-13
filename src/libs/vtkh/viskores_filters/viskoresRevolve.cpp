#include "viskoresRevolve.hpp"

#include <vtkh/Error.hpp>

#include <cmath>

#include <viskores/CellShape.h>
#include <viskores/Math.h>
#include <viskores/VectorAnalysis.h>
#include <viskores/cont/ArrayGetValues.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/CastAndCall.h>
#include <viskores/cont/CellSetExplicit.h>
#include <viskores/cont/CoordinateSystem.h>

namespace vtkh
{

namespace detail
{
inline viskores::Vec3f RotateAroundAxis(const viskores::Vec3f &p,
                                        const viskores::Vec3f &axis_unit,
                                        const viskores::Vec3f &axis_point,
                                        viskores::FloatDefault angle_radians)
{
  const viskores::FloatDefault c = static_cast<viskores::FloatDefault>(std::cos(angle_radians));
  const viskores::FloatDefault s = static_cast<viskores::FloatDefault>(std::sin(angle_radians));
  const viskores::Vec3f v = p - axis_point;
  const viskores::Vec3f k = axis_unit;
  const viskores::Vec3f v_rot = v * c + viskores::Cross(k, v) * s + k * (viskores::Dot(k, v) * (1.f - c));
  return axis_point + v_rot;
}

template <typename T, typename StorageTag>
viskores::cont::ArrayHandle<T>
Replicate(const viskores::cont::ArrayHandle<T, StorageTag> &in, viskores::Id num_repeats)
{
  const viskores::Id in_size = in.GetNumberOfValues();
  viskores::cont::ArrayHandle<T> out;
  out.Allocate(in_size * num_repeats);

  auto in_portal = in.ReadPortal();
  auto out_portal = out.WritePortal();
  for(viskores::Id r = 0; r < num_repeats; ++r)
  {
    const viskores::Id base = r * in_size;
    for(viskores::Id i = 0; i < in_size; ++i)
    {
      out_portal.Set(base + i, in_portal.Get(i));
    }
  }
  return out;
}

} // namespace detail

viskores::cont::DataSet
viskoresRevolve::Run(viskores::cont::DataSet &input,
                     viskores::filter::FieldSelection map_fields,
                     const viskores::Vec3f &axis,
                     const viskores::Vec3f &point,
                     viskores::FloatDefault angle_degrees,
                     viskores::Id num_steps,
                     bool capping)
{
  if(num_steps <= 0)
  {
    throw vtkh::Error("vtkh::Revolve num_steps must be > 0.");
  }

  (void)capping; // currently ignored (future: cap open ends)

  viskores::Vec3f axis_unit = axis;
  const viskores::FloatDefault axis_mag = viskores::Magnitude(axis_unit);
  if(axis_mag <= static_cast<viskores::FloatDefault>(0.0))
  {
    throw vtkh::Error("vtkh::Revolve axis must be non-zero.");
  }
  axis_unit = axis_unit / axis_mag;

  auto in_coords = input.GetCoordinateSystem().GetDataAsMultiplexer();

  const viskores::Id num_points = in_coords.GetNumberOfValues();
  const viskores::Id num_rings = num_steps + 1;

  viskores::cont::ArrayHandle<viskores::Vec3f> out_coords;
  out_coords.Allocate(num_points * num_rings);

  auto in_portal = in_coords.ReadPortal();
  auto out_portal = out_coords.WritePortal();

  const viskores::FloatDefault angle_radians_total =
    angle_degrees * static_cast<viskores::FloatDefault>(viskores::Pi()) / static_cast<viskores::FloatDefault>(180.0);

  for(viskores::Id ring = 0; ring < num_rings; ++ring)
  {
    const viskores::FloatDefault t =
      static_cast<viskores::FloatDefault>(ring) / static_cast<viskores::FloatDefault>(num_steps);
    const viskores::FloatDefault angle_radians = angle_radians_total * t;
    const viskores::Id ring_base = ring * num_points;

    for(viskores::Id i = 0; i < num_points; ++i)
    {
      const viskores::Vec3f p = in_portal.Get(i);
      out_portal.Set(ring_base + i, detail::RotateAroundAxis(p, axis_unit, point, angle_radians));
    }
  }

  auto cellset = input.GetCellSet();
  const viskores::Id in_num_cells = cellset.GetNumberOfCells();
  const viskores::Id out_num_cells = in_num_cells * num_steps;
  // Worst case: hexes (8 ids each).
  viskores::cont::CellSetExplicit<> out_cells;
  out_cells.PrepareToAddCells(out_num_cells, out_num_cells * 8);

  for(viskores::Id step = 0; step < num_steps; ++step)
  {
    const viskores::Id base0 = step * num_points;
    const viskores::Id base1 = (step + 1) * num_points;

    for(viskores::Id cid = 0; cid < in_num_cells; ++cid)
    {
      const viskores::IdComponent nverts = cellset.GetNumberOfPointsInCell(cid);
      viskores::Id ptids[8];
      cellset.GetCellPointIds(cid, ptids);

      if(nverts == 4)
      {
        viskores::Vec<viskores::Id, 8> hex;
        hex[0] = base0 + ptids[0];
        hex[1] = base0 + ptids[1];
        hex[2] = base0 + ptids[2];
        hex[3] = base0 + ptids[3];
        hex[4] = base1 + ptids[0];
        hex[5] = base1 + ptids[1];
        hex[6] = base1 + ptids[2];
        hex[7] = base1 + ptids[3];
        out_cells.AddCell(viskores::CELL_SHAPE_HEXAHEDRON, 8, hex);
      }
      else if(nverts == 3)
      {
        viskores::Vec<viskores::Id, 6> wedge;
        wedge[0] = base0 + ptids[0];
        wedge[1] = base0 + ptids[1];
        wedge[2] = base0 + ptids[2];
        wedge[3] = base1 + ptids[0];
        wedge[4] = base1 + ptids[1];
        wedge[5] = base1 + ptids[2];
        out_cells.AddCell(viskores::CELL_SHAPE_WEDGE, 6, wedge);
      }
      else if(nverts == 2)
      {
        viskores::Vec<viskores::Id, 4> quad;
        quad[0] = base0 + ptids[0];
        quad[1] = base0 + ptids[1];
        quad[2] = base1 + ptids[1];
        quad[3] = base1 + ptids[0];
        out_cells.AddCell(viskores::CELL_SHAPE_QUAD, 4, quad);
      }
      else
      {
        throw vtkh::Error("vtkh::Revolve only supports cells with 2, 3, or 4 vertices.");
      }
    }
  }

  out_cells.CompleteAddingCells(num_points * num_rings);

  viskores::cont::DataSet output;
  output.SetCellSet(out_cells);
  output.AddCoordinateSystem(viskores::cont::CoordinateSystem(input.GetCoordinateSystem().GetName(), out_coords));

  // Map selected fields.
  const viskores::Id num_fields = input.GetNumberOfFields();
  for(viskores::Id i = 0; i < num_fields; ++i)
  {
    const auto &f = input.GetField(i);
    if(!map_fields.IsFieldSelected(f))
    {
      continue;
    }

    if(!f.IsSupportedType())
    {
      continue;
    }

    if(f.IsPointField())
    {
      viskores::cont::CastAndCall(
        f.GetData(),
        [&](const auto &concrete) {
          auto replicated = detail::Replicate(concrete, num_rings);
          output.AddField(
            viskores::cont::Field(f.GetName(), viskores::cont::Field::Association::Points, replicated));
        });
    }
    else if(f.IsCellField())
    {
      viskores::cont::CastAndCall(
        f.GetData(),
        [&](const auto &concrete) {
          auto replicated = detail::Replicate(concrete, num_steps);
          output.AddField(
            viskores::cont::Field(f.GetName(), viskores::cont::Field::Association::Cells, replicated));
        });
    }
    else if(f.IsWholeDataSetField())
    {
      output.AddField(f);
    }
  }

  return output;
}

} // namespace vtkh
