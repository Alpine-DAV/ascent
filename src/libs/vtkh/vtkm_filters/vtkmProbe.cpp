#include "vtkmProbe.hpp"
#include <vtkm/filter/resampling/Probe.h>
#include <vtkm/cont/DataSetBuilderUniform.h>


namespace vtkh
{

using Vec2_f64 = vtkm::Vec<vtkm::Float64,2>;
using Vec3_f64 = vtkm::Vec<vtkm::Float64,3>;

//---------------------------------------------------------------------------//
vtkmProbe::vtkmProbe()
:m_mode(SampleMode::NONE)
{
  // empty
}

//---------------------------------------------------------------------------//
vtkmProbe::~vtkmProbe()
{
  // empty
}

//---------------------------------------------------------------------------//
void
vtkmProbe::setPoints(vtkm::cont::ArrayHandle<vtkm::Float64> xs,
                     vtkm::cont::ArrayHandle<vtkm::Float64> ys,
                     vtkm::cont::ArrayHandle<vtkm::Float64> zs)
{
  m_mode = SampleMode::POINTS;
  m_points_xs = xs;
  m_points_ys = ys;
  m_points_zs = zs;
}

//---------------------------------------------------------------------------//
void
vtkmProbe::setBoxDims(const Vec3_f64 dims)
{
  m_mode = SampleMode::BOX;
  m_box_dims = dims;
}

//---------------------------------------------------------------------------//
void
vtkmProbe::setBoxOrigin(const Vec3_f64 origin)
{
  m_mode = SampleMode::BOX;
  m_box_origin = origin;
}

//---------------------------------------------------------------------------//
void
vtkmProbe::setBoxSpacing(const Vec3_f64 spacing)
{
  m_mode = SampleMode::BOX;
  m_box_spacing = spacing;
}

//---------------------------------------------------------------------------//
void
vtkmProbe::setInvalidValue(const vtkm::Float64 invalid_value)
{
  m_invalid_value = invalid_value;
}

//---------------------------------------------------------------------------//
vtkm::cont::DataSet
vtkmProbe::Run(vtkm::cont::DataSet &input)
{
  vtkm::filter::resampling::Probe probe;
  vtkm::cont::DataSet ds_probe;
  if(m_mode == BOX)
  {
      if(m_box_dims[2] <= 1)
      {
        Vec2_f64 t_dims = {m_box_dims[0],m_box_dims[1]};
        Vec2_f64 t_origin = {m_box_origin[0],m_box_origin[1]};
        Vec2_f64 t_spacing = {m_box_spacing[0],m_box_spacing[1]};
        ds_probe = vtkm::cont::DataSetBuilderUniform::Create(t_dims,
                                                             t_origin,
                                                             t_spacing);
      }
      else
      {
        ds_probe = vtkm::cont::DataSetBuilderUniform::Create(m_box_dims,
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
          ds_probe.AddCoordinateSystem(vtkm::cont::CoordinateSystem("coords",
                                       make_ArrayHandleSOA(m_points_xs,
                                                           m_points_ys,
                                                           m_points_zs)));
      }
      else if(spatial_dims == 2)
      {
           vtkm::cont::ArrayHandle<vtkm::Float64> z_coords_handle;
           z_coords_handle.AllocateAndFill(m_points_xs.GetNumberOfValues(),0.0);
           ds_probe.AddCoordinateSystem(vtkm::cont::CoordinateSystem("coords",
                                       make_ArrayHandleSOA(m_points_xs,
                                                           m_points_ys,
                                                           z_coords_handle)));
      }

      vtkm::UInt8 shape_id = 1;
      vtkm::IdComponent indices_per = 1;
      vtkm::cont::CellSetSingleType<> cellset;

      // alloc conn to nverts, fill with 0 --> nverts-1)
      vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
      connectivity.Allocate(num_points);
      auto conn_portal = connectivity.WritePortal();
      for(int i = 0; i < num_points; ++i)
      {
          conn_portal.Set(i, i);
      }
      cellset.Fill(num_points, shape_id, indices_per, connectivity);
      ds_probe.SetCellSet(cellset);
  }
  else // error
  {
      std::ostringstream oss;
      oss << "vtkmProbe does not recognize sampled mode: " << m_mode;
      throw std::runtime_error(oss.str());
  }

  probe.SetGeometry(ds_probe);
  probe.SetInvalidValue(m_invalid_value);
  auto output = probe.Execute(input);
  return output;
}

} // namespace vtkh
