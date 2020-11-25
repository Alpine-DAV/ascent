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

#ifdef ASCENT_VTKM_ENABLED
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkh/filters/MarchingCubes.hpp>
#include <vtkh/rendering/RayTracer.hpp>
#include <vtkh/rendering/Scene.hpp>
#include <vtkm/rendering/Canvas.h>

#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRayTracer.h>

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

  BabelFlow::Payload serialize() const
  {
    uint32_t payl_size = sizeof(m_rangeVec);
    char* out_buffer = new char[payl_size];
    memcpy( out_buffer, (const char*)m_rangeVec, payl_size );

    return BabelFlow::Payload( payl_size, out_buffer );
  }

  void deserialize( BabelFlow::Payload payload )
  {
    memcpy( (char*)m_rangeVec, payload.buffer(), sizeof(m_rangeVec) );
  }
};


struct IsoSurfaceData
{
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

    uint32_t iso_vals_buff_size = sizeof(double) * m_IsoVals.size();
    uint32_t payl_size = 
      sizeof(vtkh::DataSet*) + 
      sizeof(size_t) + m_FieldName.size() + 
      sizeof(size_t) + iso_vals_buff_size + 
      2 * sizeof(int) + 
      sizeof(size_t) + m_FileName.size() +
      payl_bounds.size();

    char* out_buffer = new char[payl_size];
    uint32_t offset = 0;
    memcpy( out_buffer + offset, (const char*)&m_DataSet, sizeof(vtkh::DataSet*) );
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

    memcpy( out_buffer + offset, payl_bounds.buffer(), payl_bounds.size() );
    payl_bounds.reset();

    return BabelFlow::Payload( payl_size, out_buffer );
  }

  void deserialize( BabelFlow::Payload payload )
  {
    uint32_t offset = 0;
    size_t fld_name_len, num_iso_vals, file_name_len;

    memcpy( (char*)&m_DataSet, payload.buffer() + offset, sizeof(vtkh::DataSet*) );
    offset += sizeof(vtkh::DataSet*);

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

  marcher.SetInput( iso_surf_data.m_DataSet );
  marcher.SetField( iso_surf_data.m_FieldName ) ;
  marcher.SetIsoValues( iso_surf_data.m_IsoVals.data(), iso_surf_data.m_IsoVals.size() );
  marcher.Update();

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

  vtkm::Range range;
  range.Min = iso_surf_data.m_IsoVals.front();
  range.Max = iso_surf_data.m_IsoVals.back();

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
  auto& cam = render.GetCamera();
  mapper->SetCanvas( &canvas );
  mapper->RenderCells( cellset, coords, field, renderer.GetColorTable(), cam, range );

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

    m_master.initialize( m_radGatherGraph, &m_radGatherTaskMap, m_comm, &m_contMap );

    m_inputs[ BabelFlow::TaskId(m_rankId, 0) ] = m_isoSurfData.m_bounds.serialize();
    m_inputs[ BabelFlow::TaskId(m_rankId, 1) ] = m_isoSurfData.serialize();
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
  iso_surf_data.m_FileName = image_name;
  iso_surf_data.m_bounds = data.GetBounds();

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

  bflow_iso::BabelIsoGraph bflow_iso_gr( iso_surf_data, image_name, my_rank, n_ranks, mpi_comm, radix_v );
  bflow_iso_gr.Initialize();
  bflow_iso_gr.Execute();
}

