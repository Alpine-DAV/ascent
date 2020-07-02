#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>

#include <ascent_vtk_utils.hpp>
#include "ascent_runtime_babelflow_vol_utils.hpp"


#define BFLOW_VOL_UTIL_DEBUG


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin bflow_volume:: --
//-----------------------------------------------------------------------------
namespace bflow_volume
{

//-----------------------------------------------------------------------------
// -- common functions --
//-----------------------------------------------------------------------------

void split_factors(BabelFlow::TaskId id, uint32_t& x_factor, uint32_t& y_factor)
{
  BabelFlow::TaskId r = id / (BabelVolRenderingBinswap::TOTAL_NUM_BLOCKS); //round(id);

  if(r % 2 == 0)
  {
    x_factor = BabelFlow::BinarySwap::fastPow2((r/2) + 1);
    y_factor = BabelFlow::BinarySwap::fastPow2(r/2);
  }
  else
  {
    x_factor = BabelFlow::BinarySwap::fastPow2((unsigned int)std::ceil((float)r/2.f));
    y_factor = BabelFlow::BinarySwap::fastPow2((unsigned int)std::ceil((float)r/2.f));
  }

  //printf("%d: round %d factors %d %d\n", id, r, x_factor, y_factor);
}

//-----------------------------------------------------------------------------

//void split_factors_radixk(BabelFlow::TaskId id, uint32_t& x_factor, uint32_t& y_factor)
//{
//  BabelFlow::TaskId lvl = id / (BabelVolRenderingRadixK::TOTAL_NUM_BLOCKS); //round(id);
//  uint32_t split_fac = 1;
//
//  if (lvl < BabelVolRenderingRadixK::RADICES_VEC.size())   // Special case -- task is root
//  {
//    for( uint32_t i = 0; i <= lvl; ++i )
//      split_fac *= BabelVolRenderingRadixK::RADICES_VEC[i];
//  }
//
//  x_factor = 1;           // Don't split along the x-axis
//  y_factor = split_fac;   // Split only along the y-axis
//}

//-----------------------------------------------------------------------------

BabelFlow::Payload make_local_block(BabelFlow::FunctionType* data_ptr, 
                                    BabelFlow::GlobalIndexType low[3], 
                                    BabelFlow::GlobalIndexType high[3], 
                                    BabelFlow::FunctionType isovalue)
{
  BabelFlow::GlobalIndexType block_size = 
    (high[0]-low[0]+1)*(high[1]-low[1]+1)*(high[2]-low[2]+1)*sizeof(BabelFlow::FunctionType);
  BabelFlow::GlobalIndexType input_size = 
    6*sizeof(BabelFlow::GlobalIndexType) + sizeof(BabelFlow::FunctionType) + block_size;
  char* buffer = new char[input_size];

  // First - serialize data extents: low, high
  memcpy(buffer, low, 3*sizeof(BabelFlow::GlobalIndexType));
  memcpy(buffer + 3*sizeof(BabelFlow::GlobalIndexType), high, 3*sizeof(BabelFlow::GlobalIndexType));
  
  // Second - serialize the isovalue
  memcpy(buffer + 6*sizeof(BabelFlow::GlobalIndexType), &isovalue, sizeof(BabelFlow::FunctionType));
  
  // Third - serialize the data buffer
  memcpy(buffer + 6*sizeof(BabelFlow::GlobalIndexType) + sizeof(BabelFlow::FunctionType), data_ptr, block_size);

  return BabelFlow::Payload(input_size, buffer);
}

//-----------------------------------------------------------------------------

void decode_local_block(char* buffer, 
                        char*& data_ptr,
                        BabelFlow::GlobalIndexType*& low, 
                        BabelFlow::GlobalIndexType*& high, 
                        BabelFlow::FunctionType& isovalue)
{ 
  low = (BabelFlow::GlobalIndexType*)buffer;
  high = low + 3;
  isovalue = *(BabelFlow::FunctionType*)(low + 6);
  data_ptr = buffer + 6*sizeof(BabelFlow::GlobalIndexType) + sizeof(BabelFlow::FunctionType);
}

//-----------------------------------------------------------------------------

BabelFlow::Payload serialize_image(const VTKutils::ImageData& image)
{
  uint32_t zsize = (image.rend_bounds[1]-image.rend_bounds[0]+1)*(image.rend_bounds[3]-image.rend_bounds[2]+1);
  uint32_t psize = zsize*3;
  uint32_t bounds_size = 4*sizeof(uint32_t);
  uint32_t total_size = 2*bounds_size + zsize + psize;

  char* out_buffer = new char[total_size];
  memcpy(out_buffer, (const char*)image.bounds, bounds_size);
  memcpy(out_buffer + bounds_size, (const char*)image.rend_bounds, bounds_size);
  memcpy(out_buffer + 2*bounds_size, (const char*)image.zbuf, zsize);
  memcpy(out_buffer + 2*bounds_size + zsize, (const char*)image.image, psize);

  return BabelFlow::Payload(total_size, out_buffer);
}

//-----------------------------------------------------------------------------

VTKutils::ImageData deserialize_image(const char* buffer)
{
  VTKutils::ImageData image;

  unsigned char* img_buff = (unsigned char*)buffer;
  uint32_t bounds_size = 4*sizeof(uint32_t);
  
  image.bounds = (uint32_t*)img_buff;
  image.rend_bounds = (uint32_t*)(img_buff + bounds_size);

  uint32_t zsize = (image.rend_bounds[1]-image.rend_bounds[0]+1)*(image.rend_bounds[3]-image.rend_bounds[2]+1);
  uint32_t psize = 3*zsize;
  
  image.zbuf = img_buff + 2*bounds_size;
  image.image = img_buff + 2*bounds_size + zsize;

  return image;
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

  BabelFlow::GlobalIndexType *low, *high;
  BabelFlow::FunctionType isovalue;
  char* data;

  decode_local_block(inputs[0].buffer(), data, low, high, isovalue);
  
  uint32_t box_bounds[6] = {low[0], high[0], low[1], high[1], low[2], high[2]};
  VTKutils::ImageData out_image;

#ifdef BFLOW_VOL_UTIL_DEBUG    // DEBUG -- write local block extent and raw input data
  {
    std::stringstream ss;
    ss << "vol_render_" << task_id << ".params";
    std::ofstream ofs(ss.str());
    ofs << "low[0] = " << low[0] << std::endl;
    ofs << "low[1] = " << low[1] << std::endl;
    ofs << "low[2] = " << low[2] << std::endl;
    ofs << "high[0] = " << high[0] << std::endl;
    ofs << "high[1] = " << high[1] << std::endl;
    ofs << "high[2] = " << high[2] << std::endl;
    ofs.flush();
    ofs.close();
  }
  {
    BabelFlow::GlobalIndexType block_size = 
      (high[0]-low[0]+1)*(high[1]-low[1]+1)*(high[2]-low[2]+1)*sizeof(BabelFlow::FunctionType);
    std::stringstream ss;
    ss << "block_vol_render_" << task_id << ".raw";
    std::fstream fil;
    fil.open(ss.str().c_str(), std::ios::out | std::ios::binary);
    fil.write(data, block_size);
    fil.close();
  }
#endif

  VTKCompositeRender::volumeRender<BabelFlow::FunctionType>(box_bounds, data, out_image, task_id);

  outputs[0] = serialize_image(out_image);

#ifdef BFLOW_VOL_UTIL_DEBUG    // DEBUG -- write local rendering result to a file
  {
    std::stringstream filename;
    filename << "out_vol_render_" << task_id << ".png";
    VTKutils::writeImage(out_image.image, out_image.bounds, filename.str().c_str());
  }
#endif

  // Release output image memory - it was already copied into the output buffer
  delete[] out_image.zbuf;
  delete[] out_image.image;
  delete[] out_image.bounds;
  delete[] out_image.rend_bounds;

  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();
  inputs.clear();

  return 1;
}

//-----------------------------------------------------------------------------

int composite_red(std::vector<BabelFlow::Payload>& inputs, 
                  std::vector<BabelFlow::Payload>& outputs, 
                  BabelFlow::TaskId task_id)
{
  std::vector<VTKutils::ImageData> images(inputs.size());

  // The inputs are image fragments from child nodes -- read all into one array
  for(uint32_t i = 0; i < inputs.size(); ++i)
    images[i] = deserialize_image(inputs[i].buffer());
  
  // Composite all fragments
  VTKutils::ImageData out_image;
  VTKutils::composite(images, out_image, task_id);

  outputs[0] = serialize_image(out_image);

  // Release output image memory - it was already copied into the output buffer
  delete[] out_image.zbuf;
  delete[] out_image.image;
  delete[] out_image.bounds;
  delete[] out_image.rend_bounds;

  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();
  inputs.clear();

  return 1;
}

//-----------------------------------------------------------------------------

int write_results_red(std::vector<BabelFlow::Payload>& inputs,
                      std::vector<BabelFlow::Payload>& outputs, 
                      BabelFlow::TaskId task_id)
{
  // Writing results is the root of the composition reduce tree, so now we have to do
  // one last composition step
  outputs.resize(1);
  composite_red(inputs, outputs, task_id);   // will also free the memory for input buffers

  VTKutils::ImageData out_image = deserialize_image(outputs[0].buffer());
  VTKutils::writeImage(out_image.image, out_image.bounds, "final_composite.png");

  delete[] outputs[0].buffer();

  return 1;
}

//-----------------------------------------------------------------------------

int volume_render_binswap(std::vector<BabelFlow::Payload>& inputs, 
                          std::vector<BabelFlow::Payload>& outputs, 
                          BabelFlow::TaskId task_id)
{
  assert(inputs.size() == 1);

  BabelFlow::GlobalIndexType *low, *high;
  BabelFlow::FunctionType isovalue;
  char* data;

  decode_local_block(inputs[0].buffer(), data, low, high, isovalue);
  
  uint32_t box_bounds[6] = {low[0], high[0], low[1], high[1], low[2], high[2]};
  uint32_t x_factor, y_factor;
  split_factors(task_id, x_factor, y_factor);
  
  std::vector<VTKutils::ImageData> out_images;
  
  VTKCompositeRender::volumeRender<BabelFlow::FunctionType>(box_bounds, 
                                                            data, 
                                                            out_images, 
                                                            task_id, 
                                                            x_factor, 
                                                            y_factor);
  
  for(uint32_t i = 0; i < out_images.size(); ++i)
    outputs[i] = serialize_image(out_images[i]);
    
#ifdef BFLOW_VOL_UTIL_DEBUG    // DEBUG -- write local rendering result to a file
  {
    for(uint32_t i = 0; i < out_images.size(); ++i)
    {
      std::stringstream filename;
      filename << "out_vol_binswap_" << task_id << "_" << i << ".png";
      VTKutils::writeImageFixedSize(out_images[i].image, out_images[i].rend_bounds, filename.str().c_str());
    }
  }
#endif

  // Release output image memory - it was already copied into the output buffer
  for(uint32_t i = 0; i < out_images.size(); ++i)
  {
    delete[] out_images[i].zbuf;
    delete[] out_images[i].image;
    delete[] out_images[i].bounds;
    delete[] out_images[i].rend_bounds;
  }
  
  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();
  inputs.clear();
  
  return 1;
}

//-----------------------------------------------------------------------------

int composite_binswap(std::vector<BabelFlow::Payload>& inputs, 
                      std::vector<BabelFlow::Payload>& outputs, 
                      BabelFlow::TaskId task_id)
{
  std::vector<VTKutils::ImageData> images(inputs.size());

  // The inputs are image fragments from child nodes -- read all into one array
  for(uint32_t i = 0; i < inputs.size(); ++i)
    images[i] = deserialize_image(inputs[i].buffer());
    
  std::vector<VTKutils::ImageData> out_images;
  uint32_t x_factor, y_factor;
  split_factors(task_id, x_factor, y_factor);
  
  VTKutils::composite(images, out_images, task_id, x_factor, y_factor);
  
  for(uint32_t i = 0; i < out_images.size(); ++i)
    outputs[i] = serialize_image(out_images[i]);
    
#ifdef BFLOW_VOL_UTIL_DEBUG    // DEBUG -- write local rendering result to a file
  {
    for(uint32_t i = 0; i < out_images.size(); ++i)
    {
      std::stringstream filename;
      filename << "out_comp_binswap_" << task_id << "_" << i << ".png";
      VTKutils::writeImageFixedSize(out_images[i].image, out_images[i].rend_bounds, filename.str().c_str());
    }
  }
#endif

  // Release output image memory - it was already copied into the output buffer
  for(uint32_t i = 0; i < out_images.size(); ++i)
  {
    delete[] out_images[i].zbuf;
    delete[] out_images[i].image;
    delete[] out_images[i].bounds;
    delete[] out_images[i].rend_bounds;
  }
  
  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();
  inputs.clear();
  
  return 1;
}

//-----------------------------------------------------------------------------

int write_results_binswap(std::vector<BabelFlow::Payload>& inputs,
                          std::vector<BabelFlow::Payload>& outputs, 
                          BabelFlow::TaskId task_id)
{
  std::vector<VTKutils::ImageData> images(inputs.size());

  // The inputs are image fragments from child nodes -- read all into one array
  for(uint32_t i = 0; i < inputs.size(); ++i)
    images[i] = deserialize_image(inputs[i].buffer());
    
  std::vector<VTKutils::ImageData> out_images;
  uint32_t x_factor, y_factor;
  split_factors(task_id, x_factor, y_factor);
  x_factor /= 2; y_factor /= 2;
  ////
  //x_factor = 1; y_factor = 1;
  ////
  
  VTKutils::composite(images, out_images, task_id, x_factor, y_factor);
  
  uint32_t x = images[0].bounds[0];
  uint32_t y = images[0].bounds[2];
  std::stringstream filename;
  filename << "final_vol_binswap_" << x << "_" << y << ".png";
  VTKutils::writeImageFixedSize(out_images[0].image, out_images[0].rend_bounds, filename.str().c_str());
  
  // Release output image memory - it was already copied into the output buffer
  for(uint32_t i = 0; i < out_images.size(); ++i)
  {
    delete[] out_images[i].zbuf;
    delete[] out_images[i].image;
    delete[] out_images[i].bounds;
    delete[] out_images[i].rend_bounds;
  }
  
  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();
  inputs.clear();
  
  return 1;
}

//-----------------------------------------------------------------------------

int volume_render_radixk(std::vector<BabelFlow::Payload>& inputs, 
                         std::vector<BabelFlow::Payload>& outputs, 
                         BabelFlow::TaskId task_id)
{
  assert(inputs.size() == 1);

  BabelFlow::GlobalIndexType *low, *high;
  BabelFlow::FunctionType isovalue;
  char* data;

  decode_local_block(inputs[0].buffer(), data, low, high, isovalue);
  
  uint32_t box_bounds[6] = {low[0], high[0], low[1], high[1], low[2], high[2]};
  //uint32_t x_factor, y_factor;
  //split_factors_radixk(task_id, x_factor, y_factor);
  
  /////
  std::cout << "Vol render radixk, task_id = " << task_id << " num outputs = " << outputs.size() << std::endl;
  /////
  
  std::vector<VTKutils::ImageData> out_images( outputs.size() );
  
  VTKCompositeRender::volumeRenderRadixK<BabelFlow::FunctionType>(box_bounds, data, out_images, task_id);
  
  /////
  std::cout << "Vol render radixk [after render], task_id = " << task_id << " num outputs = " << outputs.size() << std::endl;
  /////
  
  for(uint32_t i = 0; i < out_images.size(); ++i)
    outputs[i] = serialize_image(out_images[i]);
    
#ifdef BFLOW_VOL_UTIL_DEBUG    // DEBUG -- write local rendering result to a file
  {
    for(uint32_t i = 0; i < out_images.size(); ++i)
    {
      std::stringstream filename;
      filename << "out_vol_radixk_" << task_id << "_" << i << ".png";
      VTKutils::writeImageFixedSize(out_images[i].image, out_images[i].rend_bounds, filename.str().c_str());
    }
  }
#endif

  // Release output image memory - it was already copied into the output buffer
  for(uint32_t i = 0; i < out_images.size(); ++i)
  {
    delete[] out_images[i].zbuf;
    delete[] out_images[i].image;
    delete[] out_images[i].bounds;
    delete[] out_images[i].rend_bounds;
  }
  
  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();
  inputs.clear();
  
  return 1;
}

//-----------------------------------------------------------------------------

int composite_radixk(std::vector<BabelFlow::Payload>& inputs, 
                     std::vector<BabelFlow::Payload>& outputs, 
                     BabelFlow::TaskId task_id)
{
  std::vector<VTKutils::ImageData> images(inputs.size());

  // The inputs are image fragments from child nodes -- read all into one array
  for(uint32_t i = 0; i < inputs.size(); ++i)
    images[i] = deserialize_image(inputs[i].buffer());
    
  std::vector<VTKutils::ImageData> out_images( outputs.size() );
  //uint32_t x_factor, y_factor;
  //split_factors(task_id, x_factor, y_factor);
  
  VTKutils::compositeRadixK(images, out_images, task_id);
  
  for(uint32_t i = 0; i < out_images.size(); ++i)
    outputs[i] = serialize_image(out_images[i]);
    
#ifdef BFLOW_VOL_UTIL_DEBUG    // DEBUG -- write local rendering result to a file
  {
    for(uint32_t i = 0; i < out_images.size(); ++i)
    {
      std::stringstream filename;
      filename << "out_comp_radixk_" << task_id << "_" << i << ".png";
      VTKutils::writeImageFixedSize(out_images[i].image, out_images[i].rend_bounds, filename.str().c_str());
    }
  }
#endif

  // Release output image memory - it was already copied into the output buffer
  for(uint32_t i = 0; i < out_images.size(); ++i)
  {
    delete[] out_images[i].zbuf;
    delete[] out_images[i].image;
    delete[] out_images[i].bounds;
    delete[] out_images[i].rend_bounds;
  }
  
  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();
  inputs.clear();
  
  return 1;
}

//-----------------------------------------------------------------------------

int write_results_radixk(std::vector<BabelFlow::Payload>& inputs,
                         std::vector<BabelFlow::Payload>& outputs, 
                         BabelFlow::TaskId task_id)
{
  // Writing results is the root of the Radix-K graph, so now we have to do
  // one last composition step
  outputs.resize(1);
  composite_radixk(inputs, outputs, task_id);   // will also free the memory for input buffers

  VTKutils::ImageData out_image = deserialize_image(outputs[0].buffer());
  uint32_t x = out_image.bounds[0];
  uint32_t y = out_image.bounds[2];
  std::stringstream filename;
  filename << "final_vol_radixk_" << x << "_" << y << ".png";
  VTKutils::writeImageFixedSize(out_image.image, out_image.rend_bounds, filename.str().c_str());
  
  delete[] outputs[0].buffer();

  return 1;
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelGraphWrapper implementation --
//-----------------------------------------------------------------------------

BabelGraphWrapper::BabelGraphWrapper(BabelFlow::FunctionType* data_ptr, 
                                     int32_t task_id, 
                                     const int32_t* data_size, 
                                     const int32_t* n_blocks,
                                     const int32_t* low, 
                                     const int32_t* high, 
                                     int32_t fanin, 
                                     BabelFlow::FunctionType extra_val, 
                                     MPI_Comm mpi_comm)
 : m_dataPtr(data_ptr), m_extraVal(extra_val), m_comm(mpi_comm)
{
  m_taskId = static_cast<uint32_t>(task_id);
  m_dataSize[0] = static_cast<uint32_t>(data_size[0]);
  m_dataSize[1] = static_cast<uint32_t>(data_size[1]);
  m_dataSize[2] = static_cast<uint32_t>(data_size[2]);
  m_nBlocks[0] = static_cast<uint32_t>(n_blocks[0]);
  m_nBlocks[1] = static_cast<uint32_t>(n_blocks[1]);
  m_nBlocks[2] = static_cast<uint32_t>(n_blocks[2]);
  m_low[0] = static_cast<BabelFlow::GlobalIndexType>(low[0]);
  m_low[1] = static_cast<BabelFlow::GlobalIndexType>(low[1]);
  m_low[2] = static_cast<BabelFlow::GlobalIndexType>(low[2]);
  m_high[0] = static_cast<BabelFlow::GlobalIndexType>(high[0]);
  m_high[1] = static_cast<BabelFlow::GlobalIndexType>(high[1]);
  m_high[2] = static_cast<BabelFlow::GlobalIndexType>(high[2]);
  m_fanin = static_cast<uint32_t>(fanin);
}

//-----------------------------------------------------------------------------

void BabelGraphWrapper::Execute()
{
  m_master.run(m_inputs);
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelVolRenderingReduce implementation --
//-----------------------------------------------------------------------------

BabelVolRenderingReduce::BabelVolRenderingReduce(BabelFlow::FunctionType* data_ptr, 
                                                 int32_t task_id, 
                                                 const int32_t* data_size, 
                                                 const int32_t* n_blocks,
                                                 const int32_t* low, 
                                                 const int32_t* high, 
                                                 int32_t fanin, 
                                                 BabelFlow::FunctionType isoval, 
                                                 MPI_Comm mpi_comm)
 : BabelGraphWrapper(data_ptr, task_id, data_size, n_blocks, low, high, fanin, isoval, mpi_comm)
{
}

//-----------------------------------------------------------------------------

void BabelVolRenderingReduce::Initialize()
{
  int mpi_size = 1;

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm_size(m_comm, &mpi_size);
#endif

  m_graph = BabelFlow::KWayReduction(m_nBlocks, m_fanin);
  m_taskMap = BabelFlow::KWayReductionTaskMap(mpi_size, &m_graph);

  m_modGraph = 
    BabelFlow::PreProcessInputTaskGraph(mpi_size, &m_graph, &m_taskMap);
  m_modMap = BabelFlow::ModTaskMap(&m_taskMap);
  m_modMap.update(m_modGraph);

  m_master.initialize(m_modGraph, &m_modMap, m_comm, &m_contMap);
  m_master.registerCallback(1, bflow_volume::volume_render_red);
  m_master.registerCallback(2, bflow_volume::composite_red);
  m_master.registerCallback(3, bflow_volume::write_results_red);
  m_master.registerCallback(m_modGraph.newCallBackId, bflow_volume::pre_proc);

  m_inputs[m_modGraph.new_tids[m_taskId]] = 
    bflow_volume::make_local_block(m_dataPtr, m_low, m_high, m_extraVal);
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelVolRenderingBinswap implementation --
//-----------------------------------------------------------------------------

uint32_t BabelVolRenderingBinswap::TOTAL_NUM_BLOCKS = 0;

BabelVolRenderingBinswap::BabelVolRenderingBinswap(BabelFlow::FunctionType* data_ptr, 
                                                   int32_t task_id, 
                                                   const int32_t* data_size, 
                                                   const int32_t* n_blocks,
                                                   const int32_t* low, 
                                                   const int32_t* high, 
                                                   int32_t fanin, 
                                                   BabelFlow::FunctionType isoval, 
                                                   MPI_Comm mpi_comm)
 : BabelGraphWrapper(data_ptr, task_id, data_size, n_blocks, low, high, fanin, isoval, mpi_comm)
{
  TOTAL_NUM_BLOCKS = m_nBlocks[0] * m_nBlocks[1] * m_nBlocks[2];
}

//-----------------------------------------------------------------------------

void BabelVolRenderingBinswap::Initialize()
{
  int mpi_size = 1;

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm_size(m_comm, &mpi_size);
#endif

  m_graph = BabelFlow::BinarySwap(m_nBlocks);
  m_taskMap = BabelFlow::BinarySwapTaskMap(mpi_size, &m_graph);

  m_modGraph = BabelFlow::PreProcessInputTaskGraph(mpi_size, &m_graph, &m_taskMap);
  m_modMap = BabelFlow::ModTaskMap(&m_taskMap);
  m_modMap.update(m_modGraph);

  m_master.initialize(m_modGraph, &m_modMap, m_comm, &m_contMap);
  m_master.registerCallback(1, bflow_volume::volume_render_binswap);
  m_master.registerCallback(2, bflow_volume::composite_binswap);
  m_master.registerCallback(3, bflow_volume::write_results_binswap);
  m_master.registerCallback(m_modGraph.newCallBackId, bflow_volume::pre_proc);

  m_inputs[m_modGraph.new_tids[m_taskId]] = 
    bflow_volume::make_local_block(m_dataPtr, m_low, m_high, m_extraVal);
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- BabelVolRenderingRadixK implementation --
//-----------------------------------------------------------------------------

uint32_t BabelVolRenderingRadixK::TOTAL_NUM_BLOCKS = 0;

std::vector<uint32_t> BabelVolRenderingRadixK::RADICES_VEC;

BabelVolRenderingRadixK::BabelVolRenderingRadixK(BabelFlow::FunctionType* data_ptr, 
                                                 int32_t task_id, 
                                                 const int32_t* data_size, 
                                                 const int32_t* n_blocks,
                                                 const int32_t* low, 
                                                 const int32_t* high, 
                                                 int32_t fanin, 
                                                 BabelFlow::FunctionType isoval,
                                                 const std::vector<uint32_t>& radix_v,
                                                 MPI_Comm mpi_comm)
 : BabelGraphWrapper(data_ptr, task_id, data_size, n_blocks, low, high, fanin, isoval, mpi_comm),
   m_Radices(radix_v)
{
  TOTAL_NUM_BLOCKS = m_nBlocks[0] * m_nBlocks[1] * m_nBlocks[2];
  RADICES_VEC = m_Radices;
}

//-----------------------------------------------------------------------------

void BabelVolRenderingRadixK::Initialize()
{
  int mpi_size = 1, myrank = 0;

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm_size(m_comm, &mpi_size);
  MPI_Comm_rank(m_comm, &myrank);
#endif

  m_graph = BabelFlow::RadixKExchange(m_nBlocks, m_Radices);
  m_taskMap = BabelFlow::RadixKExchangeTaskMap(mpi_size, &m_graph);
  
  /////
  if( myrank == 0 )
  {
    FILE* fp = fopen( "radixk.html", "w" );
    m_graph.output_graph_html( mpi_size, &m_taskMap, fp );
    fclose(fp);
  }
  /////

  m_modGraph = BabelFlow::PreProcessInputTaskGraph(mpi_size, &m_graph, &m_taskMap);
  m_modMap = BabelFlow::ModTaskMap(&m_taskMap);
  m_modMap.update(m_modGraph);
  
  /////
  if( myrank == 0 )
  {
    FILE* fp = fopen( "mod_radixk.html", "w" );
    m_modGraph.output_graph_html( mpi_size, &m_modMap, fp );
    fclose(fp);
  }
  /////

  m_master.initialize(m_modGraph, &m_modMap, m_comm, &m_contMap);
  m_master.registerCallback(1, bflow_volume::volume_render_radixk);
  m_master.registerCallback(2, bflow_volume::composite_radixk);
  m_master.registerCallback(3, bflow_volume::write_results_radixk);
  m_master.registerCallback(m_modGraph.newCallBackId, bflow_volume::pre_proc);

  m_inputs[m_modGraph.new_tids[m_taskId]] = 
    bflow_volume::make_local_block(m_dataPtr, m_low, m_high, m_extraVal);
}

//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end bflow_volume --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end ascent --
//-----------------------------------------------------------------------------



