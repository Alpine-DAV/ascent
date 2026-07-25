#include "viskoresProbe.hpp"
#include <viskores/filter/resampling/Probe.h>
#include <viskores/cont/DataSetBuilderUniform.h>


namespace vtkh
{

using Vec2_f64 = viskores::Vec<viskores::Float64,2>;
using Vec3_f64 = viskores::Vec<viskores::Float64,3>;

//---------------------------------------------------------------------------//
viskoresProbe::viskoresProbe()
:m_mode(SampleMode::NONE)
{
  // empty
}

//---------------------------------------------------------------------------//
viskoresProbe::~viskoresProbe()
{
  // empty
}

//---------------------------------------------------------------------------//
void
viskoresProbe::setPoints(viskores::cont::ArrayHandle<viskores::Float64> xs,
                     viskores::cont::ArrayHandle<viskores::Float64> ys,
                     viskores::cont::ArrayHandle<viskores::Float64> zs)
{
  m_mode = SampleMode::POINTS;
  m_points_xs = xs;
  m_points_ys = ys;
  m_points_zs = zs;
}

//---------------------------------------------------------------------------//
void
viskoresProbe::setBoxDims(const Vec3_f64 dims)
{
  m_mode = SampleMode::BOX;
  m_box_dims = dims;
}

//---------------------------------------------------------------------------//
void
viskoresProbe::setBoxOrigin(const Vec3_f64 origin)
{
  m_mode = SampleMode::BOX;
  m_box_origin = origin;
}

//---------------------------------------------------------------------------//
void
viskoresProbe::setBoxSpacing(const Vec3_f64 spacing)
{
  m_mode = SampleMode::BOX;
  m_box_spacing = spacing;
}

//---------------------------------------------------------------------------//
void
viskoresProbe::setGeometry(const viskores::cont::DataSet &geometry)
{
  m_mode = SampleMode::GEOMETRY;
  m_geometry = geometry;
}

//---------------------------------------------------------------------------//
void
viskoresProbe::setInvalidValue(const viskores::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

//---------------------------------------------------------------------------//
viskores::cont::DataSet
viskoresProbe::Run(viskores::cont::DataSet &input)
{
  viskores::filter::resampling::Probe probe;
  viskores::cont::DataSet ds_probe;
  if(m_mode == BOX)
  {
      if(m_box_dims[2] <= 1)
      {
        Vec2_f64 t_dims = {m_box_dims[0],m_box_dims[1]};
        Vec2_f64 t_origin = {m_box_origin[0],m_box_origin[1]};
        Vec2_f64 t_spacing = {m_box_spacing[0],m_box_spacing[1]};
        ds_probe = viskores::cont::DataSetBuilderUniform::Create(t_dims,
                                                             t_origin,
                                                             t_spacing);
      }
      else
      {
        ds_probe = viskores::cont::DataSetBuilderUniform::Create(m_box_dims,
                                                             m_box_origin,
                                                             m_box_spacing);
      }
  }
  else if(m_mode == POINTS)
  {
      int num_points = m_points_xs.GetNumberOfValues();
      int spatial_dims = 3;
      if( m_points_zs.GetNumberOfValues() == 0)
      {
          spatial_dims = 2;
      }
      
      if(spatial_dims == 3)
      {
          ds_probe.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",
                                       make_ArrayHandleSOA(m_points_xs,
                                                           m_points_ys,
                                                           m_points_zs)));
      }
      else if(spatial_dims == 2)
      {
           viskores::cont::ArrayHandle<viskores::Float64> z_coords_handle;
           z_coords_handle.AllocateAndFill(m_points_xs.GetNumberOfValues(),0.0);
           ds_probe.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",
                                       make_ArrayHandleSOA(m_points_xs,
                                                           m_points_ys,
                                                           z_coords_handle)));
      }

      viskores::UInt8 shape_id = 1;
      viskores::IdComponent indices_per = 1;
      viskores::cont::CellSetSingleType<> cellset;

      // alloc conn to nverts, fill with 0 --> nverts-1)
      viskores::cont::ArrayHandle<viskores::Id> connectivity;
      connectivity.Allocate(num_points);
      auto conn_portal = connectivity.WritePortal();
      for(int i = 0; i < num_points; ++i)
      {
          conn_portal.Set(i, i);
      }
      cellset.Fill(num_points, shape_id, indices_per, connectivity);
      ds_probe.SetCellSet(cellset);
  }
  else if(m_mode == GEOMETRY)
  {
      const viskores::cont::CoordinateSystem coords = m_geometry.GetCoordinateSystem();
      const viskores::Id num_points = coords.GetData().GetNumberOfValues();

      ds_probe.AddCoordinateSystem(coords);

      viskores::UInt8 shape_id = 1;
      viskores::IdComponent indices_per = 1;
      viskores::cont::CellSetSingleType<> cellset;

      viskores::cont::ArrayHandle<viskores::Id> connectivity;
      connectivity.Allocate(num_points);
      auto conn_portal = connectivity.WritePortal();
      for(viskores::Id i = 0; i < num_points; ++i)
      {
          conn_portal.Set(i, i);
      }
      cellset.Fill(num_points, shape_id, indices_per, connectivity);
      ds_probe.SetCellSet(cellset);
  }
  else // error
  {
      std::ostringstream oss;
      oss << "viskoresProbe does not recognize sampled mode: " << m_mode;
      throw std::runtime_error(oss.str());
  }

  probe.SetGeometry(ds_probe);
  probe.SetInvalidValue(m_invalid_value);
  auto output = probe.Execute(input);
  if(m_mode == GEOMETRY)
  {
      viskores::cont::DataSet topology_output;
      topology_output.CopyStructure(m_geometry);
      const viskores::Id num_fields = output.GetNumberOfFields();
      for(viskores::Id i = 0; i < num_fields; ++i)
      {
          viskores::cont::Field field = output.GetField(i);
          topology_output.AddField(viskores::cont::Field(field.GetName(),
                                                         viskores::cont::Field::Association::Points,
                                                         field.GetData()));
      }
      output = topology_output;
  }
  return output;
}

} // namespace vtkh
