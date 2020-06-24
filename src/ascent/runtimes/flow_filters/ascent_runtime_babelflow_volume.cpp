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
#include <ascent_vtk_utils.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#else
#include <mpidummy.h>
#define _NOMPI
#endif

#include "BabelFlow/TypeDefinitions.h"
#include "ascent_runtime_babelflow_filters.hpp"
#include "ascent_runtime_babelflow_vol_utils.hpp"


#define DEF_IMAGE_SIZE 1024

#define BFLOW_VOL_DEBUG


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
  res &= check_numeric("compositing", params, info, true);
  res &= check_numeric("img_size", params, info, false);
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
  conduit::Node& p = params();

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

    // Inputs for BabelFlow filter/extract -- assume 3D dataset
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
		// TODO: fix ghost cells
        //ParallelMergeTree::o_ghosts[i] = (uint32_t)in_ghosts[i];
      }
    }

    for(int i=0; i<ndims; i++)
    {
      low[i] = origin[i]/spacing[i];
      high[i] = low[i] + dims[i] -1;
      
#ifdef ASCENT_MPI_ENABLED
      MPI_Allreduce(&low[i], &global_low[i], 1, MPI_INT, MPI_MIN, color_comm);
      MPI_Allreduce(&high[i], &global_high[i], 1, MPI_INT, MPI_MAX, color_comm);
#else
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
    BabelFlow::FunctionType isovalue = p["isovalue"].as_float64();
    uint32_t img_size = DEF_IMAGE_SIZE;
    if (p.has_path("img_size"))
      img_size = static_cast<uint32_t>(p["img_size"].as_int64());
    CompositingType compositing_flag = CompositingType(p["compositing"].as_int64());

#ifdef BFLOW_VOL_DEBUG
    {
      std::stringstream ss;
      ss << "data_params_" << world_rank << ".txt";
      std::ofstream ofs(ss.str());
      ofs << "origin " << origin[0] << " " << origin[1] << " " << origin[2] << std::endl;
      ofs << "spacing " << spacing[0] << " " << spacing[1] << " " << spacing[2] << std::endl;
      ofs << "dims " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
      ofs << "low " << low[0] << " " << low[1] << " " << low[2] << std::endl;
      ofs << "high " << high[0] << " " << high[1] << " " << high[2] << std::endl;                                       
      ofs << "data_size " << data_size[0] << " " << data_size[1] << " " << data_size[2] << std::endl;
      ofs << "global_low " << global_low[0] << " " << global_low[1] << " " << global_low[2] << std::endl;
      ofs << "global_high " << global_high[0] << " " << global_high[1] << " " << global_high[2] << std::endl;
      ofs << "n_blocks " << n_blocks[0] << " " << n_blocks[1] << " " << n_blocks[2] << std::endl;
      ofs << "img_size " << img_size << std::endl;

      ofs.close();
    }
#ifdef ASCENT_MPI_ENABLED
    MPI_Barrier(color_comm);
#endif
#endif

#ifdef BFLOW_VOL_DEBUG
    {
      BabelFlow::GlobalIndexType block_size = 
        (high[0]-low[0]+1)*(high[1]-low[1]+1)*(high[2]-low[2]+1)*sizeof(BabelFlow::FunctionType);
      std::stringstream ss;
      ss << "block_" << color_rank << ".raw";
      std::fstream fil;
      fil.open(ss.str().c_str(), std::ios::out | std::ios::binary);
      fil.write((char*)array, block_size);
      fil.close();
    }
#endif

    ascent::VTKutils::setImageDims(img_size, img_size);
    ascent::VTKutils::setDatasetDims(data_size);

    switch (compositing_flag)
    {
      case CompositingType::REDUCE:
        {
          bflow_volume::BabelVolRenderingReduce red_graph(array, color_rank, data_size, n_blocks,
                                                          low, high, fanin, isovalue, color_comm);

          red_graph.Initialize();
          red_graph.Execute();
        }
        break;
      case CompositingType::BINSWAP:
        {
          bflow_volume::BabelVolRenderingBinswap binswap_graph(array, color_rank, data_size, n_blocks,
                                                               low, high, fanin, isovalue, color_comm);
          binswap_graph.Initialize();
          binswap_graph.Execute();
        }
        break;
      case CompositingType::RADIX_K:
        {
          std::vector<uint32_t> radix_v(1);
          radix_v[0] = n_blocks[0] * n_blocks[1] * n_blocks[2];
          if(p.has_path("radices"))
          {
            conduit::DataArray<int64_t> radices_arr = p["radices"].as_int64_array();
            radix_v.resize(radices_arr.number_of_elements());
            for (uint32_t i = 0; i < radix_v.size(); ++i) radix_v[i] = (uint32_t)radices_arr[i];
          }
          bflow_volume::BabelVolRenderingRadixK radixk_graph(array, color_rank, data_size, n_blocks,
                                                             low, high, fanin, isovalue, radix_v, color_comm);
          radixk_graph.Initialize();
          radixk_graph.Execute();
        }
        break;
    }
  }
}


