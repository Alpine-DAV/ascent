//
// Created by Sergei Shudler on 2020-08-21.
//

#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <mutex>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_runtime_param_check.hpp>
#include <flow_workspace.hpp>

#ifdef ASCENT_VTKM_ENABLED
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkm/rendering/Canvas.h>

#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>

#include <vtkm/cont/Serialization.h>

#include <ascent_vtkh_data_adapter.hpp>
#endif

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#define _NOMPI
#endif

#include "BabelFlow/TypeDefinitions.h"
#include "ascent_runtime_babelflow_filters.hpp"

#include "BabelFlow/charm/CharmTask.h"
#include "BabelFlow/charm/Controller.h"


#include "ascent_runtime_babelflow_comp_utils.hpp"


// #define BFLOW_ISO_DEBUG

#define DEF_IMG_WIDTH   1024
#define DEF_IMG_HEIGHT  1024


namespace BabelFlow
{
extern int relay_message(std::vector<Payload>& inputs, std::vector<Payload>& outputs, TaskId task);
}

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin bflow_comp:: --
//-----------------------------------------------------------------------------
namespace bflow_iso
{

struct BoundsData
{
  double m_rangeVec[6];   // Pairs of min, max for x, y, z axes
  double m_valRange[2];

  BoundsData& operator=( const vtkm::Bounds& bnd )
  {
    m_rangeVec[0] = bnd.X.Min;
    m_rangeVec[1] = bnd.X.Max;
    m_rangeVec[2] = bnd.Y.Min;
    m_rangeVec[3] = bnd.Y.Max;
    m_rangeVec[4] = bnd.Z.Min;
    m_rangeVec[5] = bnd.Z.Max;

    return *this;
  }

  vtkm::Bounds getVtkmBounds()
  {
    vtkm::Bounds bnd;

    bnd.X.Min = m_rangeVec[0];
    bnd.X.Max = m_rangeVec[1];
    bnd.Y.Min = m_rangeVec[2];
    bnd.Y.Max = m_rangeVec[3];
    bnd.Z.Min = m_rangeVec[4];
    bnd.Z.Max = m_rangeVec[5];

    return bnd;
  }

  vtkm::Range getVtkmRange()
  {
    vtkm::Range rng;

    rng.Min = m_valRange[0];
    rng.Max = m_valRange[1];

    return rng;
  }

  void setVtkmRange( const vtkm::Range& rng )
  {
    m_valRange[0] = rng.Min;
    m_valRange[1] = rng.Max;
  }

  BabelFlow::Payload serialize() const
  {
    uint32_t payl_size = sizeof(m_rangeVec) + sizeof(m_valRange);
    char* out_buffer = new char[payl_size];
    memcpy( out_buffer, (const char*)m_rangeVec, sizeof(m_rangeVec) );
    memcpy( out_buffer + sizeof(m_rangeVec), (const char*)m_valRange, sizeof(m_valRange) );

    return BabelFlow::Payload( payl_size, out_buffer );
  }

  void deserialize( BabelFlow::Payload payload )
  {
    memcpy( (char*)m_rangeVec, payload.buffer(), sizeof(m_rangeVec) );
    memcpy( (char*)m_valRange, payload.buffer() + sizeof(m_rangeVec), sizeof(m_valRange) );
  }
};


struct IsoSurfaceData
{
  using FieldTypeList = vtkm::List<vtkm::Float32, vtkm::Vec3f>;
  using CellSetTypes = vtkm::List<vtkm::cont::CellSetExplicit<>,
                                  vtkm::cont::CellSetSingleType<>,
                                  vtkm::cont::CellSetStructured<1>,
                                  vtkm::cont::CellSetStructured<2>,
                                  vtkm::cont::CellSetStructured<3>>;
  using DataSetWrapper = vtkm::cont::SerializableDataSet<FieldTypeList, CellSetTypes>;

  vtkh::DataSet*      m_DataSet;
  std::string         m_FieldName;
  std::vector<double> m_IsoVals;
  int                 m_Width;
  int                 m_Height;
  std::string         m_FileName;
  BoundsData          m_bounds;

  IsoSurfaceData() = default;
  
  // IsoSurfaceData( vtkh::DataSet* data_set, const std::string& fld_name, const double* iso_vals, int num_iso_vals )
  // : m_DataSet( data_set ), m_FieldName( fld_name ), m_IsoVals( iso_vals, iso_vals + num_iso_vals ) {}

  BabelFlow::Payload serialize() const
  {
    BabelFlow::Payload payl_bounds = m_bounds.serialize();

    uint32_t ds_buff_size = 0;
    size_t num_domains = m_DataSet->GetNumberOfDomains();
    std::vector<vtkmdiy::MemoryBuffer> buffs(num_domains);
    std::vector<vtkm::Id> dom_ids(num_domains);
    for( uint32_t i = 0; i < num_domains; ++i )
    {
      vtkm::cont::DataSet ds;
      m_DataSet->GetDomain(i, ds, dom_ids[i]);

      DataSetWrapper dsw(ds);
      vtkmdiy::Serialization<DataSetWrapper>::save(buffs[i], dsw);

      ds_buff_size += 
        sizeof(uint32_t) + sizeof(size_t) + buffs[i].size();
    }

    // using DataSetWrapper = vtkm::cont::SerializableDataSet<VTKM_DEFAULT_TYPE_LIST, VTKM_DEFAULT_CELL_SET_LIST>;
    // auto dsw = vtkm::cont::SerializableDataSet(data);

    // std::cout << "Mem buff size: " << buff.size() << std::endl;
    // vtkmdiy::save(buff, data);

    uint32_t iso_vals_buff_size = sizeof(double) * m_IsoVals.size();
    uint32_t payl_size = 
      // sizeof(vtkh::DataSet*) + 
      sizeof(size_t) + ds_buff_size +
      sizeof(size_t) + m_FieldName.size() + 
      sizeof(size_t) + iso_vals_buff_size + 
      2 * sizeof(int) + 
      sizeof(size_t) + m_FileName.size() +
      payl_bounds.size();

    char* out_buffer = new char[payl_size];
    uint32_t offset = 0;
    // memcpy( out_buffer + offset, (const char*)&m_DataSet, sizeof(vtkh::DataSet*) );
    // offset += sizeof(vtkh::DataSet*);
    memcpy( out_buffer + offset, (const char*)&num_domains, sizeof(size_t) );
    offset += sizeof(size_t);
    for( uint32_t i = 0; i < num_domains; ++i )
    {
      uint32_t dom_id = dom_ids[i];
      size_t sz = buffs[i].size();

      memcpy( out_buffer + offset, (const char*)&dom_id, sizeof(uint32_t) );
      offset += sizeof(uint32_t);

      // std::cout << "Serializing sz = " << sz << std::endl;

      memcpy( out_buffer + offset, (const char*)&sz, sizeof(size_t) );
      offset += sizeof(size_t);

      memcpy( out_buffer + offset, buffs[i].buffer.data(), sz );
      offset += sz;
    }

    size_t fld_name_size = m_FieldName.size();
    memcpy( out_buffer + offset, (const char*)&fld_name_size, sizeof(size_t) );
    offset += sizeof(size_t);
    memcpy( out_buffer + offset, m_FieldName.c_str(), m_FieldName.size() );
    offset += m_FieldName.size();
    size_t num_iso_vals = m_IsoVals.size();
    memcpy( out_buffer + offset, (const char*)&num_iso_vals, sizeof(size_t) );
    offset += sizeof(size_t);
    memcpy( out_buffer + offset, (const char*)m_IsoVals.data(), iso_vals_buff_size );
    offset += iso_vals_buff_size;
    memcpy( out_buffer + offset, (const char*)&m_Width, sizeof(int) );
    offset += sizeof(int);
    memcpy( out_buffer + offset, (const char*)&m_Height, sizeof(int) );
    offset += sizeof(int);
    fld_name_size = m_FileName.size();
    memcpy( out_buffer + offset, (const char*)&fld_name_size, sizeof(size_t) );
    offset += sizeof(size_t);
    memcpy( out_buffer + offset, m_FileName.c_str(), m_FileName.size() );
    offset += m_FileName.size();

    memcpy( out_buffer + offset, payl_bounds.buffer(), payl_bounds.size() );
    payl_bounds.reset();

    return BabelFlow::Payload( payl_size, out_buffer );
  }

  void deserialize( BabelFlow::Payload payload )
  {
    uint32_t offset = 0;
    size_t fld_name_len, num_iso_vals, file_name_len;

    // memcpy( (char*)&m_DataSet, payload.buffer() + offset, sizeof(vtkh::DataSet*) );
    // offset += sizeof(vtkh::DataSet*);

    m_DataSet = new vtkh::DataSet;

    size_t num_domains;
    memcpy( (char*)&num_domains, payload.buffer() + offset, sizeof(size_t) );
    offset += sizeof(size_t);

    for( uint32_t i = 0; i < num_domains; ++i )
    {
      uint32_t dom_id;
      memcpy( (char*)&dom_id, payload.buffer() + offset, sizeof(uint32_t) );
      offset += sizeof(uint32_t);

      size_t sz;
      memcpy( (char*)&sz, payload.buffer() + offset, sizeof(size_t) );
      offset += sizeof(size_t);

      vtkmdiy::MemoryBuffer membuf;
      membuf.buffer.resize(sz);
      memcpy( membuf.buffer.data(), payload.buffer() + offset, sz );
      offset += sz;

      DataSetWrapper dsw;
      vtkmdiy::Serialization<DataSetWrapper>::load(membuf, dsw);

      m_DataSet->AddDomain(dsw.DataSet, dom_id);
    }

    memcpy( (char*)(&fld_name_len), payload.buffer() + offset, sizeof(size_t) );
    offset += sizeof(size_t);

    m_FieldName.assign( payload.buffer() + offset, fld_name_len );
    offset += m_FieldName.size();

    memcpy( (char*)&num_iso_vals, payload.buffer() + offset, sizeof(size_t) );
    offset += sizeof(size_t);

    uint32_t iso_vals_buff_size = sizeof(double) * num_iso_vals;
    m_IsoVals.resize( num_iso_vals );
    memcpy( (char*)m_IsoVals.data(), payload.buffer() + offset, iso_vals_buff_size );
    offset += iso_vals_buff_size;

    memcpy( (char*)&m_Width, payload.buffer() + offset, sizeof(int) );
    offset += sizeof(int);

    memcpy( (char*)&m_Height, payload.buffer() + offset, sizeof(int) );
    offset += sizeof(int);

    memcpy( (char*)&file_name_len, payload.buffer() + offset, sizeof(size_t) );
    offset += sizeof(size_t);

    m_FileName.assign( payload.buffer() + offset, file_name_len );
    offset += m_FileName.size();

    m_bounds.deserialize( BabelFlow::Payload( payload.size() - offset, payload.buffer() + offset ) );
  }
};



int marching_cubes(std::vector<BabelFlow::Payload>& inputs, 
                   std::vector<BabelFlow::Payload>& outputs, 
                   BabelFlow::TaskId task_id)
{
  assert( inputs.size() == 2 );

  IsoSurfaceData iso_surf_data;
  iso_surf_data.deserialize( inputs[0] );

  BoundsData bounds_data;
  bounds_data.deserialize( inputs[1] );

  vtkh::MarchingCubes marcher;

  /////
  // std::cout << "Task id: " << task_id << "  marcher field name: " << iso_surf_data.m_FieldName << std::endl;
  // std::cout << "Iso vals size: " << iso_surf_data.m_IsoVals.size() << std::endl;
  // std::cout << "Field exists: " << iso_surf_data.m_DataSet->GlobalFieldExists(iso_surf_data.m_FieldName) << std::endl;
  /////

  marcher.SetInput( iso_surf_data.m_DataSet );
  marcher.SetField( iso_surf_data.m_FieldName );

  vtkm::Range range = bounds_data.getVtkmRange();

  // std::cout << "Range after allreduce: " << range.Min << "  " << range.Max << std::endl;

  int levels = int(iso_surf_data.m_IsoVals[0]);
  double step = (range.Max - range.Min) / (levels + 1.0);
  iso_surf_data.m_IsoVals.clear();
  for( int i = 1; i <= levels; ++i )
  {
    float iso = range.Min + double(i) * step;
    iso_surf_data.m_IsoVals.push_back(iso);

    // std::cout << "level [" << i << "]  " << iso << std::endl;
  }

  // bool valid_field = false;
  // bool is_cell_assoc = iso_surf_data.m_DataSet->GetFieldAssociation(iso_surf_data.m_FieldName, valid_field) ==
  //                      vtkm::cont::Field::Association::CELL_SET;
  // if(valid_field && is_cell_assoc)
  // {
  //   std::cout << "Running recenter" << std::endl;
  // }
  marcher.SetIsoValues( iso_surf_data.m_IsoVals.data(), iso_surf_data.m_IsoVals.size() );
  // marcher.SetLevels( int(iso_surf_data.m_IsoVals[0]) );

  ////
  // marcher.SetUseContourTree(true);
  ////

  marcher.Update();

  ////
  // {
  //   vtkh::Render render = vtkh::MakeRender( iso_surf_data.m_Width,
  //                                           iso_surf_data.m_Height,
  //                                           iso_surf_data.m_bounds.getVtkmBounds(),
  //                                           iso_surf_data.m_FileName );

  //   vtkh::Scene scene;
  //   vtkh::RayTracer renderer;

  //   renderer.SetInput( marcher.GetOutput() );
  //   renderer.SetField( iso_surf_data.m_FieldName );

  //   scene.AddRenderer( &renderer );
  //   scene.AddRender( render );

  //   render.GetCanvas().Clear();

  //   renderer.AddRender ( render );

  //   vtkm::cont::DataSet data_set;
  //   vtkm::Id domain_id;
  //   marcher.GetOutput()->GetDomain( 0, data_set, domain_id );
  //   const vtkm::cont::DynamicCellSet &cellset = data_set.GetCellSet();
  //   const vtkm::cont::Field &field = data_set.GetField( renderer.GetFieldName() );
  //   const vtkm::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();

  //   auto mapper = std::make_shared<vtkm::rendering::MapperRayTracer>();

  //   mapper->SetActiveColorTable( renderer.GetColorTable() );

  //   auto& canvas = render.GetCanvas();
  //   auto& cam = render.GetCamera();
  //   mapper->SetCanvas( &canvas );
  //   mapper->RenderCells( cellset, coords, field, renderer.GetColorTable(), cam, range );

  //   auto color_portal = render.GetCanvas().GetColorBuffer().ReadPortal();
  //   auto depth_portal = render.GetCanvas().GetDepthBuffer().ReadPortal();

  //   assert( color_portal.GetNumberOfValues() == iso_surf_data.m_Width*iso_surf_data.m_Height );

  //   bflow_comp::ImageData input_img;  
  //   input_img.image = new bflow_comp::ImageData::PixelType[iso_surf_data.m_Width*iso_surf_data.m_Height*bflow_comp::ImageData::sNUM_CHANNELS];
  //   input_img.zbuf = new bflow_comp::ImageData::PixelType[iso_surf_data.m_Width*iso_surf_data.m_Height];
  //   input_img.bounds = new uint32_t[4];
  //   input_img.rend_bounds = new uint32_t[4];
  //   input_img.bounds[0] = input_img.rend_bounds[0] = 0;
  //   input_img.bounds[1] = input_img.rend_bounds[1] = iso_surf_data.m_Width - 1;
  //   input_img.bounds[2] = input_img.rend_bounds[2] = 0;
  //   input_img.bounds[3] = input_img.rend_bounds[3] = iso_surf_data.m_Height - 1;

  //   uint32_t img_offset = 0;

  //   for( vtkm::Id index = 0; index < color_portal.GetNumberOfValues(); ++index )
  //   {
  //     vtkm::Vec4f_32 cur_color = color_portal.Get( index );
  //     vtkm::Float32 cur_z = depth_portal.Get( index );

  //     // input_img.image[img_offset + 0] = (unsigned char)(cur_color[0] * 255.f);
  //     // input_img.image[img_offset + 1] = (unsigned char)(cur_color[1] * 255.f);
  //     // input_img.image[img_offset + 2] = (unsigned char)(cur_color[2] * 255.f);
  //     // input_img.image[img_offset + 3] = (unsigned char)(cur_color[3] * 255.f);
  //     // input_img.zbuf[img_offset] = (unsigned char)(cur_z * 255.f);

  //     input_img.image[img_offset + 0] = cur_color[0];
  //     input_img.image[img_offset + 1] = cur_color[1];
  //     input_img.image[img_offset + 2] = cur_color[2];
  //     input_img.image[img_offset + 3] = cur_color[3];

  //     input_img.zbuf[index] = cur_z;

  //     img_offset += bflow_comp::ImageData::sNUM_CHANNELS;
  //   }

  //   std::stringstream filename;
  //   filename << task_id << ".png";
  //   input_img.writeImage(filename.str().c_str(), input_img.bounds);

  //   input_img.delBuffers();
  // }
  ////

  iso_surf_data.m_DataSet = marcher.GetOutput();
  iso_surf_data.m_bounds = bounds_data;

  outputs[0] = iso_surf_data.serialize();

  inputs[0].reset();

  return 1;
}


int vtkm_rendering(std::vector<BabelFlow::Payload>& inputs, 
                   std::vector<BabelFlow::Payload>& outputs, 
                   BabelFlow::TaskId task_id)
{
  assert( inputs.size() == 1 );

  IsoSurfaceData iso_surf_data;
  iso_surf_data.deserialize( inputs[0] );

  vtkh::Render render = vtkh::MakeRender( iso_surf_data.m_Width,
                                          iso_surf_data.m_Height,
                                          iso_surf_data.m_bounds.getVtkmBounds(),
                                          iso_surf_data.m_FileName );

  vtkh::Scene scene;
  vtkh::RayTracer renderer;

  renderer.SetInput( iso_surf_data.m_DataSet );
  renderer.SetField( iso_surf_data.m_FieldName );

  scene.AddRenderer( &renderer );
  scene.AddRender( render );

  vtkm::Range range = iso_surf_data.m_bounds.getVtkmRange();
  // range.Min = iso_surf_data.m_IsoVals.front();
  // range.Max = iso_surf_data.m_IsoVals.back();

  // {
  //   auto ranges = iso_surf_data.m_DataSet->GetGlobalRange( iso_surf_data.m_FieldName );
  //   int num_components = ranges.GetNumberOfValues();
  //   std::cout << "vtkm_rendering -- range num components = " << num_components << std::endl;
  //   vtkm::Range global_range = ranges.ReadPortal().Get(0);
  //   // a min or max may be been set by the user, check to see
  //   if(range.Min == vtkm::Infinity64())
  //   {
  //     range.Min = global_range.Min;
  //   }
  //   if(range.Max == vtkm::NegativeInfinity64())
  //   {
  //     range.Max = global_range.Max;
  //   }
  // }

#ifdef BFLOW_ISO_DEBUG
  {
    std::cout << "vtkm_rendering --" << std::endl;
    std::cout << "Num domains: " << iso_surf_data.m_DataSet->GetNumberOfDomains() << std::endl;
    std::cout << "Color table: " << renderer.GetColorTable().GetName() << std::endl;
    std::cout << "Range: " << range << std::endl;
  }
#endif
  
  render.GetCanvas().Clear();

  // renderer.SetDoComposite( true );
  renderer.AddRender ( render );

  // renderer.Update();

  vtkm::cont::DataSet data_set;
  vtkm::Id domain_id;
  iso_surf_data.m_DataSet->GetDomain( 0, data_set, domain_id );
  const vtkm::cont::DynamicCellSet &cellset = data_set.GetCellSet();
  const vtkm::cont::Field &field = data_set.GetField( renderer.GetFieldName() );
  const vtkm::cont::CoordinateSystem &coords = data_set.GetCoordinateSystem();

  auto mapper = std::make_shared<vtkm::rendering::MapperRayTracer>();

  mapper->SetActiveColorTable( renderer.GetColorTable() );

  auto& canvas = render.GetCanvas();
  auto cam = render.GetCamera();
  cam.Azimuth(-70.0);
  // cam.Elevation(-80.0);
  mapper->SetCanvas( &canvas );
  mapper->RenderCells( cellset, coords, field, renderer.GetColorTable(), cam, range );

  CkPrintf("vtkm_rendering -- finished\n");

#ifdef BFLOW_ISO_DEBUG
  {
    std::cout << "vtkm_rendering -- finished" << std::endl;
  }
#endif

  auto color_portal = render.GetCanvas().GetColorBuffer().ReadPortal();
  auto depth_portal = render.GetCanvas().GetDepthBuffer().ReadPortal();

  assert( color_portal.GetNumberOfValues() == iso_surf_data.m_Width*iso_surf_data.m_Height );

  bflow_comp::ImageData input_img;  
  input_img.image = new bflow_comp::ImageData::PixelType[iso_surf_data.m_Width*iso_surf_data.m_Height*bflow_comp::ImageData::sNUM_CHANNELS];
  input_img.zbuf = new bflow_comp::ImageData::PixelType[iso_surf_data.m_Width*iso_surf_data.m_Height];
  input_img.bounds = new uint32_t[4];
  input_img.rend_bounds = new uint32_t[4];
  input_img.bounds[0] = input_img.rend_bounds[0] = 0;
  input_img.bounds[1] = input_img.rend_bounds[1] = iso_surf_data.m_Width - 1;
  input_img.bounds[2] = input_img.rend_bounds[2] = 0;
  input_img.bounds[3] = input_img.rend_bounds[3] = iso_surf_data.m_Height - 1;

  uint32_t img_offset = 0;

  for( vtkm::Id index = 0; index < color_portal.GetNumberOfValues(); ++index )
  {
    vtkm::Vec4f_32 cur_color = color_portal.Get( index );
    vtkm::Float32 cur_z = depth_portal.Get( index );

    // input_img.image[img_offset + 0] = (unsigned char)(cur_color[0] * 255.f);
    // input_img.image[img_offset + 1] = (unsigned char)(cur_color[1] * 255.f);
    // input_img.image[img_offset + 2] = (unsigned char)(cur_color[2] * 255.f);
    // input_img.image[img_offset + 3] = (unsigned char)(cur_color[3] * 255.f);
    // input_img.zbuf[img_offset] = (unsigned char)(cur_z * 255.f);

    input_img.image[img_offset + 0] = cur_color[0];
    input_img.image[img_offset + 1] = cur_color[1];
    input_img.image[img_offset + 2] = cur_color[2];
    input_img.image[img_offset + 3] = cur_color[3];

    input_img.zbuf[index] = cur_z;

    img_offset += bflow_comp::ImageData::sNUM_CHANNELS;
  }

  // {
  //   std::stringstream filename;
  //   filename << "render_" << task_id << ".png";
  //   input_img.writeImage(filename.str().c_str(), input_img.bounds);
  // }

  // {
  //   std::stringstream filename;
  //   filename << "render_depth_" << task_id << ".png";
  //   input_img.writeDepth(filename.str().c_str(), input_img.bounds);
  // }

  outputs[0] = input_img.serialize();
  input_img.delBuffers();

  return 1;
}

int allreduce(std::vector<BabelFlow::Payload>& inputs, 
              std::vector<BabelFlow::Payload>& outputs, 
              BabelFlow::TaskId task_id)
{
  std::vector<BoundsData> bounds_v( inputs.size() );

  for( uint32_t i = 0; i < inputs.size(); ++i )
    bounds_v[i].deserialize( inputs[i] );

  BoundsData bounds_red = bounds_v[0];
  for( uint32_t i = 1; i < inputs.size(); ++i )
  {
    // X axis
    bounds_red.m_rangeVec[0] = std::min( bounds_red.m_rangeVec[0], bounds_v[i].m_rangeVec[0] );
    bounds_red.m_rangeVec[1] = std::max( bounds_red.m_rangeVec[1], bounds_v[i].m_rangeVec[1] );
    // Y axis
    bounds_red.m_rangeVec[2] = std::min( bounds_red.m_rangeVec[2], bounds_v[i].m_rangeVec[2] );
    bounds_red.m_rangeVec[3] = std::max( bounds_red.m_rangeVec[3], bounds_v[i].m_rangeVec[3] );
    // Z axis
    bounds_red.m_rangeVec[4] = std::min( bounds_red.m_rangeVec[4], bounds_v[i].m_rangeVec[4] );
    bounds_red.m_rangeVec[5] = std::max( bounds_red.m_rangeVec[5], bounds_v[i].m_rangeVec[5] );

    // Min range val
    bounds_red.m_valRange[0] = std::min( bounds_red.m_valRange[0], bounds_v[i].m_valRange[0] );
    // Max range val
    bounds_red.m_valRange[1] = std::max( bounds_red.m_valRange[1], bounds_v[i].m_valRange[1] );
  }

#ifdef BFLOW_ISO_DEBUG
  {
    std::cout << "allreduce -- task_id = " << task_id << ", inputs = " << inputs.size() << ", outputs = " << outputs.size() << std::endl;
  }
#endif

  for( uint32_t i = 0; i < outputs.size(); ++i )
    outputs[i] = bounds_red.serialize();
  
  for( BabelFlow::Payload& payl : inputs )  
    payl.reset();

  return 1;
}

class BabelIsoGraph : public bflow_comp::BabelCompRadixK
{
public:
  BabelIsoGraph( const IsoSurfaceData& iso_surf_data,
                 const std::string& img_name,
                 int32_t rank_id,
                 int32_t n_blocks,
                //  MPI_Comm mpi_comm,
                 const std::vector<uint32_t>& radix_v )
  : BabelCompRadixK( bflow_comp::ImageData(), img_name, rank_id, n_blocks, 2, radix_v ), 
    m_isoSurfData( iso_surf_data ) {}

  virtual ~BabelIsoGraph() {}

  virtual void Init(bool create_bflow_chares, int ep, int aid_v, int status_ep, int status_id)
  {
    InitRadixKGraph();
    InitGatherGraph();

    // Iso surf calc graph
    m_isoCalcTaskGr = BabelFlow::SingleTaskGraph( m_nRanks, 2, 1, 1 );
    m_isoCalcTaskMp = BabelFlow::ModuloMap( m_nRanks, m_nRanks );

    // Iso surface rendering graph
    m_isoRenderTaskGr = BabelFlow::SingleTaskGraph( m_nRanks );
    m_isoRenderTaskMp = BabelFlow::ModuloMap( m_nRanks, m_nRanks );

    m_redAllGr = BabelFlow::RadixKExchange( m_nRanks, m_Radices );
    m_redAllMp = BabelFlow::RadixKExchangeTaskMap( m_nRanks, &m_radixkGr );

    m_redAllGr.setGraphId( 0 );
    m_isoCalcTaskGr.setGraphId( 1 );
    m_isoRenderTaskGr.setGraphId( 2 );
    m_radixkGr.setGraphId( 3 );
    m_gatherTaskGr.setGraphId( 4 );

    RegisterCallbacks();

    m_isoGrConnector_1 = BabelFlow::DefGraphConnector( &m_redAllGr, 0, &m_isoCalcTaskGr, 1 );
    m_isoGrConnector_2 = BabelFlow::DefGraphConnector( &m_isoCalcTaskGr, 1, &m_isoRenderTaskGr, 2 );    
    m_isoGrConnector_3 = BabelFlow::DefGraphConnector( &m_isoRenderTaskGr, 2, &m_radixkGr, 3 );
    m_defGraphConnector = BabelFlow::DefGraphConnector( &m_radixkGr, 3, &m_gatherTaskGr, 4 );

    std::vector<BabelFlow::TaskGraphConnector*> gr_connectors{ &m_isoGrConnector_1, 
                                                               &m_isoGrConnector_2,
                                                               &m_isoGrConnector_3, 
                                                               &m_defGraphConnector };
    std::vector<BabelFlow::TaskGraph*> gr_vec{ &m_redAllGr, &m_isoCalcTaskGr, &m_isoRenderTaskGr, &m_radixkGr, &m_gatherTaskGr };
    std::vector<BabelFlow::TaskMap*> task_maps{ &m_redAllMp, &m_isoCalcTaskMp, &m_isoRenderTaskMp, &m_radixkMp, &m_gatherTaskMp };

    m_radGatherGraph = BabelFlow::ComposableTaskGraph( gr_vec, gr_connectors );
    m_radGatherTaskMap = BabelFlow::ComposableTaskMap( task_maps );

#ifdef BFLOW_ISO_DEBUG
    if( m_rankId == 0 )
    {
      m_isoCalcTaskGr.outputGraphHtml( m_nRanks, &m_isoCalcTaskMp, "iso-gr.html" );
      m_isoRenderTaskGr.outputGraphHtml( m_nRanks, &m_isoRenderTaskMp, "gather-task.html" );
      m_radixkGr.outputGraphHtml( m_nRanks, &m_radixkMp, "radixk.html" );
      // m_gatherTaskGr.outputGraphHtml( m_nRanks, &m_gatherTaskMp, "gather-task.html" );
      m_radGatherGraph.outputGraphHtml( m_nRanks, &m_radGatherTaskMap, "bflow-iso.html" );
    }
#endif

    ///
    // MPI_Barrier(m_comm);
    ///

    // m_master.initialize( m_radGatherGraph, &m_radGatherTaskMap, m_comm, &m_contMap );

    ///// Charm++
    // std::cout << "Full iso graph size: " << m_radGatherGraph.size() << std::endl;
    // m_proxy = m_controller.initialize(m_radGatherGraph.serialize(), m_radGatherGraph.size());
    if( create_bflow_chares )
    {
      BabelFlow::Payload payl = m_radGatherGraph.serialize();
      // CkPrintf("create_bflow_chares: buff size = %d  graph size = %d\n", 
      //   payl.size(), m_radGatherGraph.size());
      m_controller.initializeAsync(payl, m_radGatherGraph.size(), ep, aid_v, status_ep, status_id);
    }
    /////

    m_inputs[ BabelFlow::TaskId(m_rankId, 0) ] = m_isoSurfData.m_bounds.serialize();
    m_inputs[ BabelFlow::TaskId(m_rankId, 1) ] = m_isoSurfData.serialize();
  }

  virtual void RunChares(int bflow_chares_aid)
  {
    CkArrayID aid = CkGroupID{bflow_chares_aid};

    BabelFlow::TaskId tid_1 = BabelFlow::TaskId(m_rankId, 0);
    BabelFlow::TaskId tid_2 = BabelFlow::TaskId(m_rankId, 1);

    uint64_t gid_1 = m_radGatherGraph.gId( tid_1 );
    uint64_t gid_2 = m_radGatherGraph.gId( tid_2 );

    BabelFlow::Payload& payl_1 = m_inputs[ tid_1 ];
    std::vector<char> buffer_1( payl_1.size() );
    buffer_1.assign( payl_1.buffer(), payl_1.buffer() + payl_1.size() );

    BabelFlow::Payload& payl_2 = m_inputs[ tid_2 ];
    std::vector<char> buffer_2( payl_2.size() );
    buffer_2.assign( payl_2.buffer(), payl_2.buffer() + payl_2.size() );

    // m_proxy[m_rankId].addInput(CharmTaskId(BabelFlow::TNULL), buffer);

    BabelFlow::charm::Controller::ProxyType(aid)[gid_1].addInput(
      BabelFlow::charm::CharmTaskId(BabelFlow::TNULL), 
      buffer_1
    );

    BabelFlow::charm::Controller::ProxyType(aid)[gid_2].addInput(
      BabelFlow::charm::CharmTaskId(BabelFlow::TNULL), 
      buffer_2
    );
  }

  static void RegisterCallbacks()
  {
    BabelFlow::TaskGraph::registerCallback( 0, BabelFlow::RadixKExchange::LEAF_TASK_CB, allreduce );
    BabelFlow::TaskGraph::registerCallback( 0, BabelFlow::RadixKExchange::MID_TASK_CB, allreduce );
    BabelFlow::TaskGraph::registerCallback( 0, BabelFlow::RadixKExchange::ROOT_TASK_CB, allreduce );

    BabelFlow::TaskGraph::registerCallback( 1, BabelFlow::SingleTaskGraph::SINGLE_TASK_CB, marching_cubes );

    BabelFlow::TaskGraph::registerCallback( 2, BabelFlow::SingleTaskGraph::SINGLE_TASK_CB, vtkm_rendering );

    BabelFlow::TaskGraph::registerCallback( 3, BabelFlow::RadixKExchange::LEAF_TASK_CB, bflow_comp::volume_render_radixk );
    BabelFlow::TaskGraph::registerCallback( 3, BabelFlow::RadixKExchange::MID_TASK_CB, bflow_comp::composite_radixk );
    BabelFlow::TaskGraph::registerCallback( 3, BabelFlow::RadixKExchange::ROOT_TASK_CB, bflow_comp::composite_radixk );

    BabelFlow::TaskGraph::registerCallback( 4, BabelFlow::KWayReduction::LEAF_TASK_CB, BabelFlow::relay_message );
    BabelFlow::TaskGraph::registerCallback( 4, BabelFlow::KWayReduction::MID_TASK_CB, bflow_comp::gather_results_radixk) ;
    BabelFlow::TaskGraph::registerCallback( 4, BabelFlow::KWayReduction::ROOT_TASK_CB, bflow_comp::write_results_radixk );
  }

protected:
  IsoSurfaceData m_isoSurfData;

  BabelFlow::SingleTaskGraph m_isoCalcTaskGr;
  BabelFlow::ModuloMap m_isoCalcTaskMp;

  BabelFlow::SingleTaskGraph m_isoRenderTaskGr;
  BabelFlow::ModuloMap m_isoRenderTaskMp;

  BabelFlow::DefGraphConnector m_isoGrConnector_1;
  BabelFlow::DefGraphConnector m_isoGrConnector_2;
  BabelFlow::DefGraphConnector m_isoGrConnector_3;

  BabelFlow::RadixKExchange m_redAllGr;
  BabelFlow::RadixKExchangeTaskMap m_redAllMp; 
};


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end bflow_iso --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
///
/// BFlowIso extract
///
//-----------------------------------------------------------------------------

void ascent::runtime::filters::BFlowIso::declare_interface(conduit::Node &i)
{
  i["type_name"] = "bflow_iso";
  i["port_names"].append() = "in";
  i["output_port"] = "false";  // true -- means filter, false -- means extract
}

//-----------------------------------------------------------------------------

bool ascent::runtime::filters::BFlowIso::verify_params(const conduit::Node &params, conduit::Node &info) 
{
  info.reset();

  bool res = true;

  res &= check_string("field", params, info, true);
  res &= check_numeric("iso_values", params, info, true);
  res &= check_string("image_name", params, info, true);
  res &= check_numeric("radices", params, info, false);
  res &= check_numeric("width", params, info, false);
  res &= check_numeric("height", params, info, false);
  // res &= check_string("col_field", params, info, true);
  
  return res;
}

//-----------------------------------------------------------------------------

void ascent::runtime::filters::BFlowIso::execute() 
{
  if(!input(0).check_type<DataObject>())
  {
    ASCENT_ERROR("BFlowIso filter requires a DataObject");
  }

  conduit::Node& p = params();
  DataObject *data_object = input<DataObject>(0);

  std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();
  
  std::string image_name = p["image_name"].as_string();
  std::string field_name = p["field"].as_string();
  
  /////
  // {
  //   std::cout << "field name: " << field_name << std::endl;
  //   std::cout << "VTKHCollection fields:" << std::endl;
  //   for(auto& n : collection->field_names())
  //     std::cout << n << std::endl;

  //   std::cout << "VTKHCollection topologies:" << std::endl;
  //   for(auto& t : collection->topology_names())
  //     std::cout << t << std::endl;

  //   std::cout << "Number of children: " << data_object->as_node()->number_of_children() << std::endl;

  //   auto& data_node = data_object->as_node()->children().next();
  //   if(data_node.has_path("fields/p/values"))
  //   {
  //     int blk_x = 0, blk_y = 0, blk_z = 0;
  //     if(data_node.has_path("topologies/mesh/elements/dims/i"))
  //       blk_x = data_node["topologies/mesh/elements/dims/i"].as_int32();
  //     if(data_node.has_path("topologies/mesh/elements/dims/j"))
  //       blk_y = data_node["topologies/mesh/elements/dims/j"].as_int32();
  //     if(data_node.has_path("topologies/mesh/elements/dims/k"))
  //       blk_z = data_node["topologies/mesh/elements/dims/k"].as_int32();
      
  //     double avg = 0;
  //     conduit::DataArray<double> val_arr = 
  //       data_node["fields/p/values"].as_float64_array();
  //     for(int i = 0; i < blk_x*blk_y*blk_z; ++i)
  //       avg += val_arr[i];
  //     avg /= blk_x*blk_y*blk_z;
  //     std::cout << "Has fields p/values path, avg: " << avg << " size: " << blk_x*blk_y*blk_z << std::endl;
  //   }
  // }
  /////

  if( !collection->has_field(field_name) )
  {
    ASCENT_ERROR("BFlowIso cannot find the given field");
  }
  std::string topo_name = collection->field_topology(field_name);

  /////
  // std::cout << "Topology name: " << topo_name << "  field: " << field_name << std::endl;
  /////

  vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

  /////
  // data.PrintSummary(std::cout);
  // std::cout << "Field exists: " << data.GlobalFieldExists(field_name) << std::endl;
  /////

  /////

  /////


  bflow_iso::IsoSurfaceData iso_surf_data;
  iso_surf_data.m_DataSet = &data;
  iso_surf_data.m_FieldName = field_name;
  iso_surf_data.m_Width = DEF_IMG_WIDTH;
  iso_surf_data.m_Height = DEF_IMG_HEIGHT;
  iso_surf_data.m_FileName = image_name;
  iso_surf_data.m_bounds = data.GetBounds();  

  vtkm::cont::ArrayHandle<vtkm::Range> ranges = data.GetRange(field_name);
  vtkm::Range rng = ranges.ReadPortal().Get(0);
  // rng.Max = 500000;
  // std::cout << "Local range, min = " << rng.Min << "  max = " << rng.Max << std::endl;
  iso_surf_data.m_bounds.setVtkmRange(rng);

  if( p.has_path("width") )
  {
    iso_surf_data.m_Width = p["width"].as_int32();
  }

  if( p.has_path("height") )
  {
    iso_surf_data.m_Height = p["height"].as_int32();
  }

  // std::cout << "Width = " << iso_surf_data.m_Width << "  Height = " << iso_surf_data.m_Height << std::endl;

  const conduit::Node &n_iso_vals = p["iso_values"];
  conduit::Node n_iso_vals_dbls;
  n_iso_vals.to_float64_array(n_iso_vals_dbls);
  iso_surf_data.m_IsoVals.assign( n_iso_vals_dbls.as_double_ptr(), n_iso_vals_dbls.as_double_ptr() + n_iso_vals_dbls.dtype().number_of_elements() );

  int my_rank = 0, n_ranks = 1;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &my_rank);
  MPI_Comm_size(mpi_comm, &n_ranks);
#endif

  std::vector<uint32_t> radix_v(1);
  radix_v[0] = n_ranks;
  if( p.has_path("radices") )
  {
    conduit::DataArray<int64_t> radices_arr = p["radices"].as_int64_array();
    radix_v.resize(radices_arr.number_of_elements());
    for( uint32_t i = 0; i < radix_v.size(); ++i ) radix_v[i] = (uint32_t)radices_arr[i];

    ///
    n_ranks = radix_v[0];
    for( uint32_t i = 1; i < radix_v.size(); ++i ) n_ranks *= radix_v[i];
    ///
  }

  ///
  {
    auto& data_node = data_object->as_node()->children().next();
    if( data_node.has_path("state/domain_id") )
      my_rank = data_node["state/domain_id"].as_int32();
    // std::cout << "My rank: " << my_rank << std::endl;

    int cycle = data_node["state/cycle"].to_int();
    // std::cout << "Cycle: " << cycle << std::endl;
  }
  ///

#ifdef BFLOW_ISO_DEBUG
  {
    if( my_rank == 0 )
    {
      std::cout << ">> Iso surf data: <<" << std::endl;
      std::cout << "Field name = " << iso_surf_data.m_FieldName << std::endl;
      std::cout << "Iso vals = ";
      for( uint32_t i = 0; i < iso_surf_data.m_IsoVals.size(); ++i )
        std::cout << iso_surf_data.m_IsoVals[i] << "  ";
      std::cout << std::endl;
      std::cout << "Width = " << iso_surf_data.m_Width << std::endl;
      std::cout << "Height = " << iso_surf_data.m_Height << std::endl;
      std::cout << "File name = " << iso_surf_data.m_FileName << std::endl;
      auto ranges = iso_surf_data.m_DataSet->GetRange( iso_surf_data.m_FileName );
      int num_components = ranges.GetNumberOfValues();
      std::cout << "Num components = " << num_components << std::endl;
    }
  }
#endif

  bflow_iso::BabelIsoGraph bflow_iso_gr( iso_surf_data, image_name, my_rank, n_ranks, radix_v );

  if( p.has_path("ascent_chare_ep") )
  {
    int ep = p["ascent_chare_ep"].as_int32();
    int aid = p["ascent_chare_aid"].as_int32();

    int status_ep = p["ascent_status_ep"].as_int32();
    int status_id = p["ascent_status_aid"].as_int32();

    bflow_iso_gr.Init(true, ep, aid, status_ep, status_id);

    CkPrintf("Create bflow chares\n");
  }
  else if( p.has_path("inner_chare_aid") )
  {
    int bflow_chares_aid = p["inner_chare_aid"].as_int32();

    CkPrintf("Run bflow chares\n");

    bflow_iso_gr.Init(false, 0, 0, 0, 0);
    bflow_iso_gr.RunChares(bflow_chares_aid);
  }

  // {
  //   auto& data_node = data_object->as_node()->children().next();
  //   int rank = 0;
  //   if( data_node.has_path("state/domain_id") )
  //     rank = data_node["state/domain_id"].as_int32();
  //   std::cout << "End of iso-extract, my rank: " << rank << std::endl;
  // }
}


void register_callbacks()
{
  static std::mutex mtx;
  static bool initialized = false;

  std::unique_lock<std::mutex> lock(mtx);

  if (!initialized)
  {
    ascent::bflow_iso::BabelIsoGraph::RegisterCallbacks();
    ascent::bflow_comp::BabelGraphWrapper::sIMAGE_NAME = "iso_img";
    initialized = true;
  }
}


BabelFlow::TaskGraph* make_task_graph(BabelFlow::Payload payl)
{
  // CkPrintf("make_task_graph: size = %d\n", payl.size());
  return BabelFlow::charm::make_task_graph_template<BabelFlow::ComposableTaskGraph>(payl);
}



