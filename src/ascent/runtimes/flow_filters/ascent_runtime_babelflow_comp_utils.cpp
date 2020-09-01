//
// Created by Sergei Shudler on 2020-06-09.
//

#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>

#include <ascent_png_encoder.hpp>
#include "ascent_runtime_babelflow_comp_utils.hpp"

#include "BabelFlow/DefGraphConnector.h"
#include "BabelFlow/ComposableTaskGraph.h"
#include "BabelFlow/ComposableTaskMap.h"


#define BFLOW_COMP_UTIL_DEBUG


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin bflow_comp:: --
//-----------------------------------------------------------------------------
namespace bflow_comp
{

const uint32_t ImageData::sNUM_CHANNELS;

//-----------------------------------------------------------------------------
// -- common functions --
//-----------------------------------------------------------------------------

static inline void compute_union(const uint32_t* a,const uint32_t* b, uint32_t* c)
{
  c[0] = std::min(a[0], b[0]); c[1] = std::max(a[1], b[1]);
  c[2] = std::min(a[2], b[2]); c[3] = std::max(a[3], b[3]);
}

static inline bool compute_intersection(const uint32_t* a,const uint32_t* b, uint32_t* c)
{
  c[0] = std::max(a[0], b[0]); c[1] = std::min(a[1], b[1]);
  c[2] = std::max(a[2], b[2]); c[3] = std::min(a[3], b[3]);

  bool intersect = true;
  if( c[1] < c[0] || c[3] < c[2] ) 
  {
    c[0] = c[1] = c[2] = c[3] = 0;
    intersect = false;
  }

  return intersect;
}

//-----------------------------------------------------------------------------

void ImageData::writeImage(const char* filename, uint32_t* extent)
{
  int x_extent = extent[1] - extent[0] + 1;
  int y_extent = extent[3] - extent[2] + 1;
  
  // PNGEncoder works with rgba -- 4 channels
  unsigned char* pixel_buff = new unsigned char[x_extent*y_extent*4]();
  
  uint32_t x_size = rend_bounds[1] - rend_bounds[0] + 1;
  uint32_t y_size = rend_bounds[3] - rend_bounds[2] + 1;
  
  for(int y=0; y < y_extent; ++y)
  {
      for(int x=0; x < x_extent; ++x)
      {
        int idx = x + y*x_extent; 
        int imgidx = idx*4;
        
        if( (x + extent[0] < rend_bounds[0]) || (x + extent[0] > rend_bounds[1]) || 
            (y + extent[2] < rend_bounds[2]) || (y + extent[2] > rend_bounds[3]) )
        {
          continue;
        }
        
        uint32_t myidx = (x + extent[0] -rend_bounds[0]) + (y + extent[2] - rend_bounds[2])*x_size;
        uint32_t myimgidx = myidx*ImageData::sNUM_CHANNELS;
        
        pixel_buff[imgidx + 3] = 255;
        memcpy(pixel_buff + imgidx, image + myimgidx, ImageData::sNUM_CHANNELS);
      }
  }

  // Use Ascent's PNGEncoder to write image to disk
  PNGEncoder encoder;
  encoder.Encode(pixel_buff, x_extent, y_extent);
  encoder.Save(filename);
  
  delete[] pixel_buff;
}

BabelFlow::Payload ImageData::serialize() const
{
  uint32_t zsize = (rend_bounds[1]-rend_bounds[0]+1)*(rend_bounds[3]-rend_bounds[2]+1);
  uint32_t psize = zsize*ImageData::sNUM_CHANNELS;
  uint32_t bounds_size = 4*sizeof(uint32_t);
  uint32_t total_size = 2*bounds_size + zsize + psize;

  char* out_buffer = new char[total_size];
  memcpy(out_buffer, (const char*)bounds, bounds_size);
  memcpy(out_buffer + bounds_size, (const char*)rend_bounds, bounds_size);
  memcpy(out_buffer + 2*bounds_size, (const char*)zbuf, zsize);
  memcpy(out_buffer + 2*bounds_size + zsize, (const char*)image, psize);

  return BabelFlow::Payload(total_size, out_buffer);
}

void ImageData::deserialize(BabelFlow::Payload payload)
{
  unsigned char* img_buff = (unsigned char*)payload.buffer();
  uint32_t bounds_size = 4*sizeof(uint32_t);
  
  bounds = (uint32_t*)img_buff;
  rend_bounds = (uint32_t*)(img_buff + bounds_size);

  uint32_t zsize = (rend_bounds[1]-rend_bounds[0]+1)*(rend_bounds[3]-rend_bounds[2]+1);
  uint32_t psize = ImageData::sNUM_CHANNELS*zsize;
  
  zbuf = img_buff + 2*bounds_size;
  image = img_buff + 2*bounds_size + zsize;
}

void ImageData::delBuffers()
{
  delete[] zbuf; zbuf = nullptr;
  delete[] image; image = nullptr;
  delete[] bounds; bounds = nullptr;
  delete[] rend_bounds; rend_bounds = nullptr;
}

//-----------------------------------------------------------------------------

void split_and_blend(const std::vector<ImageData>& input_images,
                     std::vector<ImageData>& out_images,
                     uint32_t* union_box,
                     bool flip_split_side,
                     bool skip_z_check,
                     bool union_box_as_extent)
{
  uint32_t extent[4];
  if( union_box_as_extent )
  {
    memcpy(extent, union_box, 4*sizeof(uint32_t));
#ifdef BFLOW_COMP_UTIL_DEBUG
    std::cout << "union_box: " << union_box[0] << " "
                               << union_box[1] << " "
                               << union_box[2] << " "
                               << union_box[3] << std::endl;
#endif
  }
  else
  {
    memcpy(extent, input_images[0].bounds, 4*sizeof(uint32_t));
  }
  
  uint32_t x_size = extent[1] - extent[0] + 1;
  uint32_t y_size = extent[3] - extent[2] + 1;
  
  uint32_t split_size[2] = {x_size / (uint32_t)out_images.size(), y_size / (uint32_t)out_images.size()};
  
  // Split factor is the size of the out_images array
  int split_dir = 1;    // 0 -- x-axis, 1 -- y-axis
  // TODO: fractions
  if( flip_split_side && x_size >= y_size )
    split_dir = 0;
  
  for( int i = 0; i < out_images.size(); ++i )
  {
    ImageData& outimg = out_images[i];
    
    outimg.bounds = new uint32_t[4];
    outimg.rend_bounds = new uint32_t[4];
    
    if( split_dir == 0 )    // Split along x-axis
    {
      outimg.bounds[0] = extent[0] + i*split_size[0];
      outimg.bounds[1] = extent[0] + (i+1)*split_size[0] - 1; 
      outimg.bounds[2] = extent[2];
      outimg.bounds[3] = extent[3];
    }
    else                    // Split along y-axis
    {
      outimg.bounds[0] = extent[0];
      outimg.bounds[1] = extent[1]; 
      outimg.bounds[2] = extent[2] + i*split_size[1];
      outimg.bounds[3] = extent[2] + (i+1)*split_size[1] - 1;
    }

    compute_intersection( outimg.bounds, union_box, outimg.rend_bounds );

    uint32_t zsize = 
      (outimg.rend_bounds[1] - outimg.rend_bounds[0] + 1) * (outimg.rend_bounds[3] - outimg.rend_bounds[2] + 1);
    outimg.image = new unsigned char[zsize*ImageData::sNUM_CHANNELS]();
    // Initialize alpha channel to 255 (fully opaque)
    if( ImageData::sNUM_CHANNELS > 3 )
    {
      for( uint32_t j = 0; j < zsize; ++j )
        outimg.image[j*ImageData::sNUM_CHANNELS + 3] = 255;
    }
    outimg.zbuf = new unsigned char[zsize]();
    
    if( zsize == 0 )
      std::cout << i << " image is empty" << std::endl;
  }
  
  for( uint32_t j = 0; j < input_images.size(); ++j )      // Blend every input image
  {
    const ImageData& inimg = input_images[j];
    
    uint32_t x_size = inimg.rend_bounds[1] - inimg.rend_bounds[0] + 1;
    
    for( uint32_t i = 0; i < out_images.size(); ++i )
    {
      ImageData& outimg = out_images[i];
      
      uint32_t* bound = outimg.rend_bounds;

      for( uint32_t y = bound[2]; y < bound[3] + 1; ++y )
      {
        for( uint32_t x = bound[0]; x < bound[1] + 1; ++x )
        {
          if( x < inimg.rend_bounds[0] || x > inimg.rend_bounds[1] || 
              y < inimg.rend_bounds[2] || y > inimg.rend_bounds[3] )
          {
            //std::cout << "split_and_blend, shouldn't be here, x = " << x << ", y = " << y << std::endl;
            continue;
          }
          
          uint32_t rx = x - inimg.rend_bounds[0];
          uint32_t ry = y - inimg.rend_bounds[2];

          uint32_t idx = (rx + ry*x_size);
          uint32_t imgidx = idx*ImageData::sNUM_CHANNELS;

          uint32_t myidx = (x - bound[0]) + (y - bound[2]) * (bound[1] - bound[0] + 1);
          uint32_t imgmyidx = myidx*ImageData::sNUM_CHANNELS;

          if( skip_z_check || outimg.zbuf[myidx] < inimg.zbuf[idx] )
          {
            outimg.zbuf[myidx] = inimg.zbuf[idx];
            memcpy(outimg.image + imgmyidx, inimg.image + imgidx, ImageData::sNUM_CHANNELS);
          }
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------

void compose_images( const std::vector<ImageData>& in_images, 
                     std::vector<ImageData>& out_images, 
                     int id,
                     bool flip_split_side,
                     bool skip_z_check,
                     bool union_box_as_extent )
{
  uint32_t union_box[4];
  memcpy( union_box, in_images[0].rend_bounds, 4*sizeof(uint32_t) );

  for( uint32_t i = 1; i < in_images.size(); ++i )
  {
    bflow_comp::compute_union( union_box, in_images[i].rend_bounds, union_box );
  }

  bflow_comp::split_and_blend( in_images, out_images, union_box, flip_split_side, skip_z_check, union_box_as_extent );
}


//-----------------------------------------------------------------------------

int generic_composite(std::vector<BabelFlow::Payload>& inputs, 
                      std::vector<BabelFlow::Payload>& outputs, 
                      BabelFlow::TaskId task_id,
                      bool flip_split_side,
                      bool skip_z_check,
                      bool union_box_as_extent)
{
  std::vector<ImageData> images(inputs.size());

  // The inputs are image fragments from child nodes -- read all into one array
  for(uint32_t i = 0; i < inputs.size(); ++i)
    images[i].deserialize(inputs[i]);
  
  std::vector<ImageData> out_images(outputs.size());
  
  bflow_comp::compose_images(images, out_images, task_id, flip_split_side, skip_z_check, union_box_as_extent);
  
  for(uint32_t i = 0; i < out_images.size(); ++i)
  {
    outputs[i] = out_images[i].serialize();
    
#ifdef BFLOW_COMP_UTIL_DEBUG    // DEBUG -- write local rendering result to a file
    {
      std::stringstream filename;
      filename << "composite_" << task_id << "_" << i << ".png";
      out_images[i].writeImage(filename.str().c_str(), out_images[i].rend_bounds);
    }
#endif
    
    out_images[i].delBuffers();
  }
  
  for( BabelFlow::Payload& payl : inputs )  
    payl.reset();
  
  return 1;
}

//-----------------------------------------------------------------------------

int pre_proc(std::vector<BabelFlow::Payload>& inputs, 
             std::vector<BabelFlow::Payload>& outputs, 
             BabelFlow::TaskId task_id)
{
  outputs = inputs;
  
  return 1;
}

//-----------------------------------------------------------------------------

int volume_render_red(std::vector<BabelFlow::Payload>& inputs, 
                      std::vector<BabelFlow::Payload>& outputs, 
                      BabelFlow::TaskId task_id)
{
  assert(inputs.size() == 1);
  
  outputs = inputs;

  return 1;
}

//-----------------------------------------------------------------------------

int composite_red(std::vector<BabelFlow::Payload>& inputs, 
                  std::vector<BabelFlow::Payload>& outputs, 
                  BabelFlow::TaskId task_id)
{
  // Compositing -- do not flip split direction and do a zbuf check
  return generic_composite(inputs, outputs, task_id, false, false, false);
}

//-----------------------------------------------------------------------------

int write_results_red(std::vector<BabelFlow::Payload>& inputs,
                      std::vector<BabelFlow::Payload>& outputs, 
                      BabelFlow::TaskId task_id)
{
  // Writing results is the root of the composition reduce tree, so now we have to do
  // one last composition step
  std::vector<BabelFlow::Payload> comp_outputs( 1 );
  composite_red(inputs, comp_outputs, task_id);   // will also free the memory for input buffers

  ImageData out_image;
  out_image.deserialize( comp_outputs[0] );
  std::stringstream filename;
  filename << BabelGraphWrapper::sIMAGE_NAME << ".png";
  out_image.writeImage(filename.str().c_str(), out_image.bounds);

  comp_outputs[0].reset();

  return 1;
}

//-----------------------------------------------------------------------------

int volume_render_binswap(std::vector<BabelFlow::Payload>& inputs, 
                          std::vector<BabelFlow::Payload>& outputs, 
                          BabelFlow::TaskId task_id)
{
  assert(inputs.size() == 1);
  
  // Compositing -- flip the split direction (as in traditional bin-swap) and 
  // skip the zbuf check (only one input)
  return generic_composite(inputs, outputs, task_id, true, true, false);
}

//-----------------------------------------------------------------------------

int composite_binswap(std::vector<BabelFlow::Payload>& inputs, 
                      std::vector<BabelFlow::Payload>& outputs, 
                      BabelFlow::TaskId task_id)
{
  // Compositing -- flip the split direction (as in traditional bin-swap) and
  // do the zbuf check
  return generic_composite(inputs, outputs, task_id, true, false, false);
}

//-----------------------------------------------------------------------------

int write_results_binswap(std::vector<BabelFlow::Payload>& inputs,
                          std::vector<BabelFlow::Payload>& outputs, 
                          BabelFlow::TaskId task_id)
{
  // Writing results is the root of the binswap graph, so now we have to do
  // one last composition step
  std::vector<BabelFlow::Payload> comp_outputs( 1 );
  composite_binswap(inputs, comp_outputs, task_id);   // will also free the memory for input buffers
  
  ImageData out_image;
  out_image.deserialize( comp_outputs[0] );
  uint32_t x = out_image.bounds[0];
  uint32_t y = out_image.bounds[2];
  std::stringstream filename;
  filename << BabelGraphWrapper::sIMAGE_NAME << "_" << x << "_" << y << ".png";
  out_image.writeImage(filename.str().c_str(), out_image.rend_bounds);
  
  comp_outputs[0].reset();
  
  return 1;
}

//-----------------------------------------------------------------------------

int volume_render_radixk(std::vector<BabelFlow::Payload>& inputs, 
                         std::vector<BabelFlow::Payload>& outputs, 
                         BabelFlow::TaskId task_id)
{
  assert(inputs.size() == 1);

  // Compositing -- don't flip the split direction (as in traditional k-radix) and 
  // skip the zbuf check (only one input)
  return generic_composite(inputs, outputs, task_id, false, true, false);
}

//-----------------------------------------------------------------------------

int composite_radixk(std::vector<BabelFlow::Payload>& inputs, 
                     std::vector<BabelFlow::Payload>& outputs, 
                     BabelFlow::TaskId task_id)
{
  // Compositing -- don't flip the split direction (as in traditional k-radix) and 
  // do the zbuf check
  return generic_composite(inputs, outputs, task_id, false, false, false);
}

//-----------------------------------------------------------------------------

int write_results_radixk(std::vector<BabelFlow::Payload>& inputs,
                         std::vector<BabelFlow::Payload>& outputs, 
                         BabelFlow::TaskId task_id)
{
  // Writing results is the root of the Radix-K graph, so now we have to do
  // one last composition step
  std::vector<BabelFlow::Payload> comp_outputs( 1 );
  // will also free the memory for input buffers
  generic_composite(inputs, comp_outputs, task_id, false, false, true);

  ImageData out_image;
  out_image.deserialize( comp_outputs[0] );
  std::stringstream filename;
  filename << BabelGraphWrapper::sIMAGE_NAME << ".png";
  out_image.writeImage(filename.str().c_str(), out_image.rend_bounds);
  
  comp_outputs[0].reset();

  return 1;
}

//-----------------------------------------------------------------------------

int gather_results_radixk(std::vector<BabelFlow::Payload>& inputs,
                          std::vector<BabelFlow::Payload>& outputs, 
                          BabelFlow::TaskId task_id)
{
  return generic_composite( inputs, outputs, task_id, false, false, true );
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelGraphWrapper implementation --
//-----------------------------------------------------------------------------

std::string BabelGraphWrapper::sIMAGE_NAME;

BabelGraphWrapper::BabelGraphWrapper(const ImageData& input_img,
                                     const std::string& img_name,
                                     int32_t rank_id,
                                     int32_t n_ranks,
                                     int32_t fanin,
                                     MPI_Comm mpi_comm)
 : m_inputImg( input_img ), m_comm( mpi_comm )
{
  m_rankId = static_cast<uint32_t>( rank_id );
  m_nRanks = static_cast<uint32_t>( n_ranks );
  m_fanin = static_cast<uint32_t>( fanin );
  
  sIMAGE_NAME = img_name;
}

//-----------------------------------------------------------------------------

void BabelGraphWrapper::Execute()
{
  m_master.run(m_inputs);
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelCompReduce implementation --
//-----------------------------------------------------------------------------

BabelCompReduce::BabelCompReduce(const ImageData& input_image,
                                 const std::string& img_name,
                                 int32_t rank_id,
                                 int32_t n_blocks,
                                 int32_t fanin,
                                 MPI_Comm mpi_comm)
 : BabelGraphWrapper( input_image, img_name, rank_id, n_blocks, fanin, mpi_comm )
{
}

//-----------------------------------------------------------------------------

void BabelCompReduce::Initialize()
{
  uint32_t blks[3] = { m_nRanks, 1, 1 };
  m_graph = BabelFlow::KWayReduction( blks, m_fanin );
  m_graph.registerCallback( BabelFlow::KWayReduction::LEAF_TASK_CB, bflow_comp::volume_render_red );
  m_graph.registerCallback( BabelFlow::KWayReduction::MID_TASK_CB, bflow_comp::composite_red) ;
  m_graph.registerCallback( BabelFlow::KWayReduction::ROOT_TASK_CB, bflow_comp::write_results_red );

  m_taskMap = BabelFlow::KWayReductionTaskMap( m_nRanks, &m_graph );

  m_modGraph = BabelFlow::PreProcessInputTaskGraph( m_nRanks, &m_graph, &m_taskMap );
  m_modGraph.registerCallback( BabelFlow::PreProcessInputTaskGraph::PRE_PROC_TASK_CB, bflow_comp::pre_proc );

  m_modMap = BabelFlow::ModTaskMap( &m_taskMap );
  m_modMap.update( m_modGraph );

#ifdef BFLOW_COMP_UTIL_DEBUG
  if( m_rankId == 0 )
  {
    m_graph.outputGraphHtml( m_nRanks, &m_taskMap, "reduce.html" );
  }
#endif

  m_master.initialize( m_modGraph, &m_modMap, m_comm, &m_contMap );

  m_inputs[m_modGraph.new_tids[m_rankId]] = m_inputImg.serialize();
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelVolRenderingBinswap implementation --
//-----------------------------------------------------------------------------

BabelCompBinswap::BabelCompBinswap(const ImageData& input_image,
                                   const std::string& img_name,
                                   int32_t rank_id,
                                   int32_t n_blocks,
                                   int32_t fanin,
                                   MPI_Comm mpi_comm)
 : BabelGraphWrapper(input_image, img_name, rank_id, n_blocks, fanin, mpi_comm)
{
}

//-----------------------------------------------------------------------------

void BabelCompBinswap::Initialize()
{
  m_graph = BabelFlow::BinarySwap( m_nRanks );
  m_graph.registerCallback( BabelFlow::BinarySwap::LEAF_TASK_CB, bflow_comp::volume_render_binswap );
  m_graph.registerCallback( BabelFlow::BinarySwap::MID_TASK_CB, bflow_comp::composite_binswap );
  m_graph.registerCallback( BabelFlow::BinarySwap::ROOT_TASK_CB, bflow_comp::write_results_binswap );

  m_taskMap = BabelFlow::BinarySwapTaskMap( m_nRanks, &m_graph );

#ifdef BFLOW_COMP_UTIL_DEBUG
  if( m_rankId == 0 )
  {
    m_graph.outputGraphHtml( m_nRanks, &m_taskMap, "bin-swap.html" );
  }
#endif

  m_modGraph = BabelFlow::PreProcessInputTaskGraph( m_nRanks, &m_graph, &m_taskMap );
  m_modGraph.registerCallback( BabelFlow::PreProcessInputTaskGraph::PRE_PROC_TASK_CB, bflow_comp::pre_proc );

  m_modMap = BabelFlow::ModTaskMap( &m_taskMap );
  m_modMap.update( m_modGraph );

  m_master.initialize( m_modGraph, &m_modMap, m_comm, &m_contMap );

  m_inputs[m_modGraph.new_tids[m_rankId]] = m_inputImg.serialize();
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelVolRenderingRadixK implementation --
//-----------------------------------------------------------------------------

BabelCompRadixK::BabelCompRadixK(const ImageData& input_image,
                                 const std::string& img_name,
                                 int32_t rank_id,
                                 int32_t n_blocks,
                                 int32_t fanin,
                                 MPI_Comm mpi_comm,
                                 const std::vector<uint32_t>& radix_v)
 : BabelGraphWrapper(input_image, img_name, rank_id, n_blocks, fanin, mpi_comm), m_Radices(radix_v)
{
}

//-----------------------------------------------------------------------------

BabelCompRadixK::~BabelCompRadixK()
{
}

//-----------------------------------------------------------------------------

void BabelCompRadixK::InitRadixKGraph()
{
  // RadixK exchange graph
  m_radixkGr = BabelFlow::RadixKExchange( m_nRanks, m_Radices );
  m_radixkGr.registerCallback( BabelFlow::RadixKExchange::LEAF_TASK_CB, bflow_comp::volume_render_radixk );
  m_radixkGr.registerCallback( BabelFlow::RadixKExchange::MID_TASK_CB, bflow_comp::composite_radixk );
  m_radixkGr.registerCallback( BabelFlow::RadixKExchange::ROOT_TASK_CB, bflow_comp::composite_radixk );
  m_radixkMp = BabelFlow::RadixKExchangeTaskMap( m_nRanks, &m_radixkGr );
}

//-----------------------------------------------------------------------------

void BabelCompRadixK::InitGatherGraph()
{
  // Gather graph
  uint32_t blks[3] = { m_nRanks, 1, 1 };
  m_gatherTaskGr = BabelFlow::KWayReduction( blks, m_fanin );
  m_gatherTaskGr.registerCallback( BabelFlow::KWayReduction::LEAF_TASK_CB, bflow_comp::pre_proc );
  m_gatherTaskGr.registerCallback( BabelFlow::KWayReduction::MID_TASK_CB, bflow_comp::gather_results_radixk) ;
  m_gatherTaskGr.registerCallback( BabelFlow::KWayReduction::ROOT_TASK_CB, bflow_comp::write_results_radixk );
  m_gatherTaskMp = BabelFlow::KWayReductionTaskMap( m_nRanks, &m_gatherTaskGr );
}

//-----------------------------------------------------------------------------

void BabelCompRadixK::Initialize()
{
  InitRadixKGraph();
  InitGatherGraph();
  
  /////
  // Pre-process graph
  // m_preProcTaskGr = BabelFlow::SingleTaskGraph();
  // m_preProcTaskGr.registerCallback( BabelFlow::SingleTaskGraph::SINGLE_TASK_CB, bflow_comp::pre_proc );
  // m_preProcTaskMp = BabelFlow::ModuloMap( m_nRanks, m_nRanks );
  /////

  /////
  // m_defGraphConnectorPreProc = BabelFlow::DefGraphConnector( m_nRanks,
  //                                                            &m_preProcTaskGr, 0,
  //                                                            &m_radixkGr, 1,
  //                                                            &m_preProcTaskMp,
  //                                                            &m_radixkMp );
  /////

  m_defGraphConnector = BabelFlow::DefGraphConnector( m_nRanks,
                                                      &m_radixkGr, 0,
                                                      &m_gatherTaskGr, 1,
                                                      &m_radixkMp,
                                                      &m_gatherTaskMp );

  // std::vector<BabelFlow::TaskGraphConnector*> gr_connectors{ &m_defGraphConnectorPreProc, &m_defGraphConnector };
  // std::vector<BabelFlow::TaskGraph*> gr_vec{ &m_preProcTaskGr, &m_radixkGr, &m_gatherTaskGr };
  // std::vector<BabelFlow::TaskMap*> task_maps{ &m_preProcTaskMp, &m_radixkMp, &m_gatherTaskMp }; 

  std::vector<BabelFlow::TaskGraphConnector*> gr_connectors{ &m_defGraphConnector };
  std::vector<BabelFlow::TaskGraph*> gr_vec{ &m_radixkGr, &m_gatherTaskGr };
  std::vector<BabelFlow::TaskMap*> task_maps{ &m_radixkMp, &m_gatherTaskMp }; 

  m_radGatherGraph = BabelFlow::ComposableTaskGraph( gr_vec, gr_connectors );
  m_radGatherTaskMap = BabelFlow::ComposableTaskMap( task_maps );
  
#ifdef BFLOW_COMP_UTIL_DEBUG
  if( m_rankId == 0 )
  {
    m_preProcTaskGr.outputGraphHtml( m_nRanks, &m_preProcTaskMp, "pre-proc.html" );
    m_radixkGr.outputGraphHtml( m_nRanks, &m_radixkMp, "radixk.html" );
    m_gatherTaskGr.outputGraphHtml( m_nRanks, &m_gatherTaskMp, "gather-task.html" );
    m_radGatherGraph.outputGraphHtml( m_nRanks, &m_radGatherTaskMap, "radixk-gather.html" );
  }
#endif

  m_master.initialize( m_radGatherGraph, &m_radGatherTaskMap, m_comm, &m_contMap );

  m_inputs[m_rankId] = m_inputImg.serialize();
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end bflow_comp --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent --
//-----------------------------------------------------------------------------



