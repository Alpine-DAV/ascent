
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_runtime_param_check.hpp>
#include <flow_workspace.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif

#include "BabelFlow/TypeDefinitions.h"
#include "BabelFlow/mpi/Controller.h"
#include "BabelFlow/vlr/KWayReduction.h"
#include "BabelFlow/vlr/KWayReductionTaskMap.h"
#include "BabelFlow/PreProcessInputTaskGraph.hpp"
#include "BabelFlow/ModTaskMap.hpp"

#include "ascent_vtk_utils.h"
#include "ascent_runtime_babelflow_filters.hpp"


//-----------------------------------------------------------------------------
// -- begin bflow_volume:: --
//-----------------------------------------------------------------------------
namespace bflow_volume
{

class BabelVolumeRendering
{
public:
  BabelVolumeRendering(BabelFlow::FunctionType* data_ptr, int32_t task_id, 
                       const int32_t* data_size, const int32_t* n_blocks,
                       const int32_t* low, const int32_t* high, int32_t fanin, 
                       BabelFlow::FunctionType isovalue, MPI_Comm mpi_comm);

  void Initialize();

  void Execute();


private:
  BabelFlow::FunctionType* m_dataPtr;
  uint32_t m_taskId;
  uint32_t m_dataSize[3];
  uint32_t m_nBlocks[3];
  BabelFlow::GlobalIndexType m_low[3];
  BabelFlow::GlobalIndexType m_high[3];
  uint32_t m_fanin;
  BabelFlow::FunctionType m_isovalue;
  MPI_Comm m_comm;

  std::map<BabelFlow::TaskId, BabelFlow::Payload> m_inputs;

  BabelFlow::mpi::Controller m_master;
  BabelFlow::ControllerMap m_contMap;
  BabelFlow::KWayReduction m_graph;
  BabelFlow::KWayReductionTaskMap m_taskMap; 

  BabelFlow::PreProcessInputTaskGraph<BabelFlow::KWayReduction> m_modGraph;
  BabelFlow::ModTaskMap<BabelFlow::KWayReductionTaskMap> m_modMap;
};

//-----------------------------------------------------------------------------

BabelFlow::Payload make_local_block(BabelFlow::FunctionType* data_ptr, BabelFlow::GlobalIndexType low[3], 
                                    BabelFlow::GlobalIndexType high[3], BabelFlow::FunctionType isovalue)
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

BabelFlow::Payload serialize_image(const VTKutils::SimpleImageData& image)
{
  uint32_t img_size = (image.bounds[1]-image.bounds[0]+1)*(image.bounds[3]-image.bounds[2]+1);
  uint32_t zsize = img_size;
  uint32_t psize = img_size*3;
  uint32_t bounds_size = 4*sizeof(uint32_t);

  char* out_buffer = new char[bounds_size + zsize + psize];
  memcpy(out_buffer, (const char*)image.bounds, bounds_size);
  memcpy(out_buffer + bounds_size, (const char*)image.zbuf, zsize);
  memcpy(out_buffer + bounds_size + zsize, (const char*)image.image, psize);

  return Payload(bounds_size + zsize + psize, out_buffer);
}

//-----------------------------------------------------------------------------

VTKutils::SimpleImageData deserialize_image(const char* buffer)
{
  VTKutils::SimpleImageData image;

  unsigned char* img_buff = (unsigned char*)buffer;
  image.bounds = (uint32_t*)img_buff;

  uint32_t img_size = (image.bounds[1]-image.bounds[0]+1)*(image.bounds[3]-image.bounds[2]+1);
  uint32_t zsize = img_size;
  uint32_t psize = 3*img_size;

  image.zbuf = img_buff + bounds_size;
  image.image = img_buff + bounds_size + zsize;

  return image;
}

//-----------------------------------------------------------------------------

int pre_proc(std::vector<BabelFlow::Payload>& inputs, 
             std::vector<BabelFlow::Payload>& output, BabelFlow::TaskId task_id)
{

  return 1;
}

//-----------------------------------------------------------------------------

int volume_render(std::vector<BabelFlow::Payload>& inputs, 
                  std::vector<BabelFlow::Payload>& output, BabelFlow::TaskId task_id)
{
  assert(inputs.size() == 1);

  BabelFlow::GlobalIndexType* low = (BabelFlow::GlobalIndexType*)inputs[0].buffer();
  BabelFlow::GlobalIndexType* high = (BabelFlow::GlobalIndexType*)(inputs[0].buffer()) + 3;
  BabelFlow::FunctionType isovalue = 
    *(BabelFlow::FunctionType*)((BabelFlow::GlobalIndexType*)(inputs[0].buffer()) + 6);
  char* data = 
    inputs[0].buffer() + 6*sizeof(BabelFlow::GlobalIndexType) + sizeof(BabelFlow::FunctionType);

  uint32_t box_bounds[6] = {low[0], high[0], low[1], high[1], low[2], high[2]};
  VTKutils::SimpleImageData out_image;

  VTKCompositeRender::volumeRender(box_bounds, data, isovalue, out_image, task_id);

  outputs[0] = serialize_image(out_image);

#ifdef 0    // DEBUG -- write local rendering result to a file
  {
    std::stringstream filename;
    filename << "out_vol_render_" << task_id << ".png";
    VTKutils utils;
    utils.arraytoImage(out_image.image, out_image.bounds, filename.str());
  }
#endif

  // Release output image memory - it was already copied into the output buffer
  delete[] out_image.zbuf;
  delete[] out_image.image;
  delete[] out_image.bounds;

  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();

  return 1;
}

//-----------------------------------------------------------------------------

int composite(std::vector<BabelFlow::Payload>& inputs, 
              std::vector<BabelFlow::Payload>& output, BabelFlow::TaskId task_id)
{
  VTKutils utils;
  std::vector<VTKutils::SimpleImageData> images(inputs.size());
  uint32_t bounds_size = 4*sizeof(uint32_t);

  // The inputs are image fragments from child nodes -- read all into one array
  for(uint32_t i = 0; i < inputs.size(); ++i)
    images[i] = deserialize_image(inputs[i].buffer());

  // Composite all fragments
  VTKutils::SimpleImageData out_image;
  VTKCompositeRender::composite(images, out_image, task_id);

  outputs[0] = serialize_image(out_image);

  // Release output image memory - it was already copied into the output buffer
  delete[] out_image.zbuf;
  delete[] out_image.image;
  delete[] out_image.bounds;

  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();

  return 1;
}

//-----------------------------------------------------------------------------

int write_results(std::vector<BabelFlow::Payload>& inputs,
                  std::vector<BabelFlow::Payload>& output, BabelFlow::TaskId task_id)
{
  // Writing results is the root of the composition reduce tree, so now we have to do
  // one last composition step
  outputs.resize(1);
  composite(inputs, outputs, task_id);

  VTKutils::SimpleImageData out_image = deserialize_image(outputs[0].buffer());
  VTKCompositeRender::write_image(out_image, out_image.bounds, "final_composite.png");

  for(uint32_t i = 0; i < inputs.size(); ++i)
    delete[] inputs[i].buffer();

  delete[] outputs[0].buffer();

  return 1;
}

//-----------------------------------------------------------------------------

BabelVolumeRendering::BabelVolumeRendering(BabelFlow::FunctionType* data_ptr, int32_t task_id, 
                                           const int32_t* data_size, const int32_t* n_blocks,
                                           const int32_t* low, const int32_t* high, int32_t fanin, 
                                           BabelFlow::FunctionType isovalue, MPI_Comm mpi_comm)
 : m_dataPtr(data_ptr), m_isovalue(isovalue), m_comm(mpi_comm)
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

void BabelVolumeRendering::Initialize()
{
  int my_rank = 0;
  int mpi_size = 1;

#ifdef ASCENT_MPI_ENABLED
  MPI_Comm_rank(m_comm, &my_rank);
  MPI_Comm_size(m_comm, &mpi_size);
#endif

  m_graph = BabelFlow::KWayReduction(m_nBlocks, m_fanin);
  m_taskMap = BabelFlow::KWayReductionTaskMap(mpi_size, &m_graph);

  m_modGraph = BabelFlow::PreProcessInputTaskGraph<BabelFlow::KWayReduction>(mpi_size, &m_graph, &m_taskMap);
  m_modMap = BabelFlow::ModTaskMap<KWayTaskMap>(&m_taskMap);
  m_modMap.update(m_modGraph);

  m_master.initialize(m_modGraph, &m_modMap, m_comm, &m_contMap);
  m_master.registerCallback(1, bflow_volume::volume_render);
  m_master.registerCallback(2, bflow_volume::composite);
  m_master.registerCallback(3, bflow_volume::write_results);
  m_master.registerCallback(m_modGraph.newCallBackId, bflow_volume::pre_proc);

  inputs[m_modGraph.new_tids[m_taskId]] = bflow_volume::make_local_block(m_dataPtr, m_low, m_high, m_isovalue);
}

//-----------------------------------------------------------------------------

void BabelVolumeRendering::Execute()
{
  m_master.run(m_inputs);
}


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end bflow_volume:: --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
///
/// BFlowVolume Filter
///
//-----------------------------------------------------------------------------

void ascent::runtime::filters::BFlowVolume::declare_interface(conduit::Node &i)
{
  i["type_name"] = "bflow_volume";
  i["port_names"].append() = "in";
  i["output_port"] = "false";  // true -- means filter, false -- means extract
}

//-----------------------------------------------------------------------------

bool ascent::runtime::filters::BFlowVolume::verify_params(const conduit::Node &params, conduit::Node &info) 
{
  info.reset();

  bool res = true;

  res &= check_string("field", params, info, true);
  res &= check_numeric("fanin", params, info, true);
  res &= check_numeric("isovalue", params, info, true);
  res &= check_numeric("ugrid_select", params, info, false);
  
  return res;
}

//-----------------------------------------------------------------------------

void ascent::runtime::filters::BFlowVolume::execute() 
{
  if(!input(0).check_type<DataObject>())
  {
      ASCENT_ERROR("BFlowVolume filter requires a DataObject");
  }

  // connect to the input port and get the parameters
  DataObject *d_input = input<DataObject>(0);  
  conduit::Node& data_node = d_input->as_node()->children().next();
  conduit::Node p = params();

  int color = 0;
  int uniform_color = 0;

  // check if coordset uniform
  if(data_node.has_path("coordsets/coords/type"))
  {
    std::string coordSetType = data_node["coordsets/coords/type"].as_string();
    uniform_color = (coordSetType == "uniform") ? 1 : 0;
  }
  else
    ASCENT_ERROR("BabelFlow filter could not find coordsets/coords/type");

  // Decide which uniform grid to work on (default 0, the finest spacing)
  int32_t selected_spacing = 0;

  int world_rank = 0, uniform_rank = 0, uniform_comm_size = 1;
#ifdef ASCENT_MPI_ENABLED
  MPI_Comm world_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(world_comm, &world_rank);

  MPI_Comm uniform_comm;
  MPI_Comm_split(world_comm, uniform_color, world_rank, &uniform_comm);

  MPI_Comm_rank(uniform_comm, &uniform_rank);
  MPI_Comm_size(uniform_comm, &uniform_comm_size);
#endif

  if(uniform_color)
  {
    int32_t myspacing = 0;
    
    // uniform grid should not have spacing as {x,y,z}
    // this is a workaround to support old Ascent dataset using {x,y,z}
    if(data_node.has_path("coordsets/coords/spacing/x"))
      myspacing = data_node["coordsets/coords/spacing/x"].value();
    else if(data_node.has_path("coordsets/coords/spacing/dx"))
      myspacing = data_node["coordsets/coords/spacing/dx"].value();
    
    std::vector<int32_t> uniform_spacing(uniform_comm_size);

#ifdef ASCENT_MPI_ENABLED
    MPI_Allgather(&myspacing, 1, MPI_INT, uniform_spacing.data(), 1, MPI_INT, uniform_comm);
#endif
    
    std::sort(uniform_spacing.begin(), uniform_spacing.end());
    std::unique(uniform_spacing.begin(), uniform_spacing.end());
    
    if(p.has_path("ugrid_select")) 
      selected_spacing = *std::next(uniform_spacing.begin(), p["ugrid_select"].as_int64());
    else
      selected_spacing = *std::next(uniform_spacing.begin(), 0);
    
    color = (myspacing == selected_spacing);
    
    //std::cout << "Selected spacing "<< selected_spacing << " rank " << world_rank << " contributing " << color <<"\n";
  }

#ifdef ASCENT_MPI_ENABLED
  MPI_Barrier(uniform_comm);

  MPI_Comm color_comm;
  MPI_Comm_split(uniform_comm, color, uniform_rank, &color_comm);

  int color_rank, color_comm_size;
  MPI_Comm_rank(color_comm, &color_rank);
  MPI_Comm_size(color_comm, &color_comm_size);
#endif

  conduit::Node& fields_root_node = data_node["fields"];
  conduit::Node& field_node = fields_root_node[p["field"].as_string()];

  conduit::DataArray<double> array_mag = field_node["values"].as_float64_array();

  if(color) 
  {
    const int ndims = data_node.has_path("coordsets/coords/dims/k") ? 3 : 2;

    // NOTE: when field is a vector the coords/spacing has dx/dy/dz
    int32_t dims[3] = {1, 1, 1};
    int32_t spacing[3] = {1, 1, 1};
    int32_t origin[3] = {0, 0, 0};
    
    dims[0] = data_node["coordsets/coords/dims/i"].value();
    dims[1] = data_node["coordsets/coords/dims/j"].value();
    if(ndims > 2)
      dims[2] = data_node["coordsets/coords/dims/k"].value();

    if(data_node.has_path("coordsets/coords/spacing"))
    {
      if(data_node.has_path("coordsets/coords/spacing/x"))
      {
        spacing[0] = data_node["coordsets/coords/spacing/x"].value();
        spacing[1] = data_node["coordsets/coords/spacing/y"].value();
        if(ndims > 2)
          spacing[2] = data_node["coordsets/coords/spacing/z"].value();

        data_node["coordsets/coords/spacing/dx"] = spacing[0];
        data_node["coordsets/coords/spacing/dy"] = spacing[1];
        data_node["coordsets/coords/spacing/dz"] = spacing[2];
      }
      else if(data_node.has_path("coordsets/coords/spacing/dx"))
      {
        spacing[0] = data_node["coordsets/coords/spacing/dx"].value();
        spacing[1] = data_node["coordsets/coords/spacing/dy"].value();
        if(ndims > 2)
          spacing[2] = data_node["coordsets/coords/spacing/dz"].value();
      }
    }

    origin[0] = data_node["coordsets/coords/origin/x"].value();
    origin[1] = data_node["coordsets/coords/origin/y"].value();
    if(ndims > 2)
      origin[2] = data_node["coordsets/coords/origin/z"].value();

    // Inputs of PMT assume 3D dataset
    int32_t low[3] = {0,0,0};
    int32_t high[3] = {0,0,0};

    int32_t global_low[3] = {0,0,0};
    int32_t global_high[3] = {0,0,0};
    int32_t data_size[3] = {1,1,1};

    int32_t n_blocks[3] = {1,1,1};

    if(p.has_path("in_ghosts")) 
    {
      int64_t* in_ghosts = p["in_ghosts"].as_int64_ptr();
      for(int i=0;i< 6;i++) {
        ParallelMergeTree::o_ghosts[i] = (uint32_t)in_ghosts[i];
      }
    }

    for(int i=0; i<ndims; i++)
    {
      low[i] = origin[i]/spacing[i];
      high[i] = low[i] + dims[i] -1;
      
#ifdef ASCENT_MPI_ENABLED
      MPI_Allreduce(&low[i], &global_low[i], 1, MPI_INT, MPI_MIN, color_comm);
      MPI_Allreduce(&high[i], &global_high[i], 1, MPI_INT, MPI_MAX, color_comm);
#elif
      global_low[i] = low[i];
      global_high[i] = high[i];
#endif
      data_size[i] = global_high[i]-global_low[i]+1;
      // normalize box
      low[i] -= global_low[i];
      high[i] = low[i] + dims[i] -1;

      n_blocks[i] = std::ceil(data_size[i]*1.0/dims[i]);
    }

    // get the data handle
    BabelFlow::FunctionType* array = reinterpret_cast<BabelFlow::FunctionType *>(array_mag.data_ptr());

    int64_t fanin = p["fanin"].as_int64();
    FunctionType isovalue = p["isovalue"].as_float64();
    int64_t gen_field = p["gen_segment"].as_int64();

    VTKutils::set_dataset_dims(data_size);

    BabelVolumeRendering vlr(array, color_rank, data_size, n_blocks,
                             low, high, fanin, isovalue, color_comm);

    vlr.Initialize();
    vlr.Execute();
  }
}


