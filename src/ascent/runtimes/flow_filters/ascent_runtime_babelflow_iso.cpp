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

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_runtime_param_check.hpp>
#include <flow_workspace.hpp>

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkm/rendering/Canvas.h>

#include <ascent_vtkh_data_adapter.hpp>
#endif

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif

#include "BabelFlow/TypeDefinitions.h"
#include "ascent_runtime_babelflow_filters.hpp"
#include "ascent_runtime_babelflow_comp_utils.hpp"


//#define BFLOW_ISO_DEBUG
#define DEF_IMG_WIDTH   1024
#define DEF_IMG_HEIGHT  1024


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


struct IsoSurfaceData
{
  vtkh::DataSet*      m_DataSet;
  std::string         m_FieldName;
  std::vector<double> m_IsoVals;
  int                 m_Width;
  int                 m_Height;
  std::string         m_FileName;

  IsoSurfaceData() = default;
  
  // IsoSurfaceData( vtkh::DataSet* data_set, const std::string& fld_name, const double* iso_vals, int num_iso_vals )
  // : m_DataSet( data_set ), m_FieldName( fld_name ), m_IsoVals( iso_vals, iso_vals + num_iso_vals ) {}

  BabelFlow::Payload serialize() const
  {
    uint32_t iso_vals_buff_size = sizeof(double) * m_IsoVals.size();
    uint32_t payl_size = 
      sizeof(vtkh::DataSet*) + 
      sizeof(size_t) + 
      m_FieldName.size() + 
      sizeof(size_t) + 
      iso_vals_buff_size + 
      2 * sizeof(int) + 
      sizeof(size_t) + 
      m_FileName.size();

    char* out_buffer = new char[payl_size];
    uint32_t offset = 0;
    memcpy( out_buffer + offset, (const char*)m_DataSet, sizeof(vtkh::DataSet*) );
    offset += sizeof(vtkh::DataSet*);
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

    return BabelFlow::Payload( payl_size, out_buffer );
  }

  void deserialize( BabelFlow::Payload payload )
  {
    uint32_t offset = 0;
    size_t fld_name_len, num_iso_vals, file_name_len;

    memcpy( (char*)m_DataSet, payload.buffer() + offset, sizeof(vtkh::DataSet*) );
    offset += sizeof(vtkh::DataSet*);

    memcpy( (char*)(&fld_name_len), payload.buffer() + offset, sizeof(size_t) );
    offset += sizeof(size_t);

    m_FieldName.assign( payload.buffer() + offset, fld_name_len );
    offset += m_FieldName.size();
    // char* fld_name = new char[fld_name_len + 1];
    // memcpy( fld_name, payload.buffer() + sizeof(vtkh::DataSet*) + sizeof(size_t), fld_name_len );
    // fld_name[fld_name_len] = '\0';
    // m_FieldName = fld_name;
    // delete[] fld_name;

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
  }
};



int marching_cubes(std::vector<BabelFlow::Payload>& inputs, 
                   std::vector<BabelFlow::Payload>& outputs, 
                   BabelFlow::TaskId task_id)
{
  assert( inputs.size() == 1 );
  
  IsoSurfaceData iso_surf_data;
  iso_surf_data.deserialize( inputs[0] );

  vtkh::MarchingCubes marcher;

  marcher.SetInput( iso_surf_data.m_DataSet );
  marcher.SetField( iso_surf_data.m_FieldName) ;
  marcher.SetIsoValues( iso_surf_data.m_IsoVals.data(), iso_surf_data.m_IsoVals.size() );
  marcher.Update();

  iso_surf_data.m_DataSet = marcher.GetOutput();

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
                                          iso_surf_data.m_DataSet->GetBounds(),
                                          iso_surf_data.m_FileName );

  vtkh::Scene scene;
  vtkh::RayTracer renderer;
  // vtkh::Renderer *renderer = nullptr;

  renderer.SetField( iso_surf_data.m_FieldName );

  scene.AddRenderer( &renderer );
  scene.AddRender( render );
  scene.Render();

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

    input_img.image[img_offset + 0] = (unsigned char)(cur_color[0] * 255.f);
    input_img.image[img_offset + 1] = (unsigned char)(cur_color[1] * 255.f);
    input_img.image[img_offset + 2] = (unsigned char)(cur_color[2] * 255.f);
    input_img.image[img_offset + 3] = (unsigned char)(cur_color[3] * 255.f);

    input_img.zbuf[img_offset] = (unsigned char)(cur_z * 255.f);

    img_offset += bflow_comp::ImageData::sNUM_CHANNELS;
  }

  outputs[0] = input_img.serialize();
  input_img.delBuffers();

  return 1;
}

class BabelIsoGraph : public bflow_comp::BabelCompRadixK
{
public:
  BabelIsoGraph( const IsoSurfaceData& iso_surf_data,
                 const std::string& img_name,
                 int32_t rank_id,
                 int32_t n_blocks,
                 MPI_Comm mpi_comm,
                 const std::vector<uint32_t>& radix_v )
  : BabelCompRadixK( bflow_comp::ImageData(), img_name, rank_id, n_blocks, 2, mpi_comm, radix_v ), 
    m_isoSurfData( iso_surf_data ) {}

  virtual ~BabelIsoGraph() {}

  virtual void Initialize() override
  {
    InitRadixKGraph();
    InitGatherGraph();

    // Iso surf calc graph
    m_isoCalcTaskGr = BabelFlow::SingleTaskGraph();
    m_isoCalcTaskGr.registerCallback( BabelFlow::SingleTaskGraph::SINGLE_TASK_CB, marching_cubes );
    m_isoCalcTaskMp = BabelFlow::ModuloMap( m_nRanks, m_nRanks );

    // Iso surface rendering graph
    m_isoRenderTaskGr = BabelFlow::SingleTaskGraph();
    m_isoRenderTaskGr.registerCallback( BabelFlow::SingleTaskGraph::SINGLE_TASK_CB, vtkm_rendering );
    m_isoRenderTaskMp = BabelFlow::ModuloMap( m_nRanks, m_nRanks );

    m_isoGrConnector_1 = BabelFlow::DefGraphConnector( m_nRanks,
                                                     &m_isoCalcTaskGr, 0,
                                                     &m_isoRenderTaskGr, 1,
                                                     &m_isoCalcTaskMp,
                                                     &m_isoRenderTaskMp );

    m_isoGrConnector_2 = BabelFlow::DefGraphConnector( m_nRanks,
                                                       &m_isoRenderTaskGr, 1,
                                                       &m_radixkGr, 2,
                                                       &m_isoRenderTaskMp,
                                                       &m_radixkMp );

    m_defGraphConnector = BabelFlow::DefGraphConnector( m_nRanks,
                                                      &m_radixkGr, 2,
                                                      &m_gatherTaskGr, 3,
                                                      &m_radixkMp,
                                                      &m_gatherTaskMp );

    std::vector<BabelFlow::TaskGraphConnector*> gr_connectors{ &m_isoGrConnector_1, &m_isoGrConnector_2, &m_defGraphConnector };
    std::vector<BabelFlow::TaskGraph*> gr_vec{ &m_isoCalcTaskGr, &m_isoRenderTaskGr, &m_radixkGr, &m_gatherTaskGr };
    std::vector<BabelFlow::TaskMap*> task_maps{ &m_isoCalcTaskMp, &m_isoRenderTaskMp, &m_radixkMp, &m_gatherTaskMp };

    m_radGatherGraph = BabelFlow::ComposableTaskGraph( gr_vec, gr_connectors );
    m_radGatherTaskMap = BabelFlow::ComposableTaskMap( task_maps );

#ifdef BFLOW_ISO_DEBUG
    if( m_rankId == 0 )
    {
      // m_radixkGr.outputGraphHtml( m_nRanks, &m_radixkMp, "radixk.html" );
      // m_gatherTaskGr.outputGraphHtml( m_nRanks, &m_gatherTaskMp, "gather-task.html" );
      m_radGatherGraph.outputGraphHtml( m_nRanks, &m_radGatherTaskMap, "bflow-iso.html" );
    }
#endif

    m_master.initialize( m_radGatherGraph, &m_radGatherTaskMap, m_comm, &m_contMap );

    m_inputs[m_rankId] = m_isoSurfData.serialize();
  }

protected:
  IsoSurfaceData m_isoSurfData;

  BabelFlow::SingleTaskGraph m_isoCalcTaskGr;
  BabelFlow::ModuloMap m_isoCalcTaskMp;

  BabelFlow::SingleTaskGraph m_isoRenderTaskGr;
  BabelFlow::ModuloMap m_isoRenderTaskMp;

  BabelFlow::DefGraphConnector m_isoGrConnector_1;
  BabelFlow::DefGraphConnector m_isoGrConnector_2;
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
  if( !collection->has_field(field_name) )
  {
    ASCENT_ERROR("BFlowIso cannot find the given field");
  }
  std::string topo_name = collection->field_topology(field_name);
  vtkh::DataSet &data = collection->dataset_by_topology(topo_name);

  bflow_iso::IsoSurfaceData iso_surf_data;
  iso_surf_data.m_DataSet = &data;
  iso_surf_data.m_FieldName = field_name;
  iso_surf_data.m_Width = DEF_IMG_WIDTH;
  iso_surf_data.m_Height = DEF_IMG_HEIGHT;

  if( p.has_path("width") )
  {
    iso_surf_data.m_Width = p["width"].as_int64();
  }

  if( p.has_path("height") )
  {
    iso_surf_data.m_Height = p["height"].as_int64();
  }

  const conduit::Node &n_iso_vals = p["iso_values"];
  conduit::Node n_iso_vals_dbls;
  n_iso_vals.to_float64_array(n_iso_vals_dbls);
  iso_surf_data.m_IsoVals.assign( n_iso_vals_dbls.as_double_ptr(), n_iso_vals_dbls.as_double_ptr() + n_iso_vals_dbls.dtype().number_of_elements() );

  MPI_Comm mpi_comm;
  int my_rank = 0, n_ranks = 1;
#ifdef ASCENT_MPI_ENABLED
  mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
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
  }

  bflow_iso::BabelIsoGraph bflow_iso_gr( iso_surf_data, image_name, my_rank, n_ranks, mpi_comm, radix_v );
  bflow_iso_gr.Initialize();
  bflow_iso_gr.Execute();
}

