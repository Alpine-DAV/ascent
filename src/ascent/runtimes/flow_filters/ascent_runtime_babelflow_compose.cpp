//
// Created by Sergei Shudler on 2020-06-09.
//

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
#include "ascent_runtime_babelflow_filters.hpp"
#include "ascent_runtime_babelflow_comp_utils.hpp"


//#define BFLOW_COMP_DEBUG


//-----------------------------------------------------------------------------
///
/// BFlowCompose Filter
///
//-----------------------------------------------------------------------------

void ascent::runtime::filters::BFlowCompose::declare_interface(conduit::Node &i)
{
  i["type_name"] = "bflow_comp";
  i["port_names"].append() = "in";
  i["output_port"] = "false";  // true -- means filter, false -- means extract
}

//-----------------------------------------------------------------------------

bool ascent::runtime::filters::BFlowCompose::verify_params(const conduit::Node &params, conduit::Node &info) 
{
  info.reset();

  bool res = true;

  res &= check_string("color_field", params, info, true);
  res &= check_string("depth_field", params, info, true);
  res &= check_string("image_name", params, info, true);
  res &= check_numeric("compositing", params, info, true);
  res &= check_numeric("fanin", params, info, false);
  
  return res;
}

//-----------------------------------------------------------------------------

void ascent::runtime::filters::BFlowCompose::execute() 
{
  if(!input(0).check_type<DataObject>())
  {
      ASCENT_ERROR("BFlowVolume filter requires a DataObject");
  }

  // Connect to the input port and get the parameters
  DataObject *d_input = input<DataObject>(0);  
  conduit::Node& data_node = d_input->as_node()->children().next();
  conduit::Node& p = params();

  // Check if coordset valid, meaning: uniform and with spacing of 1
  if( !data_node.has_path("coordsets/coords/type") ||
      data_node["coordsets/coords/type"].as_string() != "uniform" )
  {
     ASCENT_ERROR("BabelFlow comp extract could not find coordsets/coords/type or type is not 'uniform'");
  }

  if( data_node.has_path("coordsets/coords/spacing/dx") &&
      data_node["coordsets/coords/spacing/dx"].as_int32() != 1 )
  {
    ASCENT_ERROR("BabelFlow comp extract requires spacing of 1 along the X-axis");
  }
  
  if( data_node.has_path("coordsets/coords/spacing/dy") &&
      data_node["coordsets/coords/spacing/dy"].as_int32() != 1 )
  {
    ASCENT_ERROR("BabelFlow comp extract requires spacing of 1 along the Y-axis");
  }
  
  if( data_node["coordsets/coords/origin/x"].as_int32() != 0 ||
      data_node["coordsets/coords/origin/y"].as_int32() != 0 )
  {
    ASCENT_ERROR("BabelFlow comp extract requires origin (0,0)");
  }
  
  // Width and height reduced by 1 because of 'element' association
  int32_t img_width  = data_node["coordsets/coords/dims/i"].as_int32() - 1;
  int32_t img_height = data_node["coordsets/coords/dims/j"].as_int32() - 1;
  
  conduit::Node& fields_root_node = data_node["fields"];
  conduit::Node& color_node = fields_root_node[p["color_field"].as_string()];
  conduit::Node& depth_node = fields_root_node[p["depth_field"].as_string()];
  
  if( color_node["association"].as_string() != "element" ||
      depth_node["association"].as_string() != "element" )
  {
    ASCENT_ERROR("BabelFlow comp extract requires element association in pixel and zbuf fields");
  }
  
  // Convert pixel and zbuf data to unsigned char arrays
  conduit::Node converted_pixels;
  color_node["values"].to_unsigned_char_array(converted_pixels);
  conduit::DataArray<unsigned char> pixels_arr = converted_pixels.as_unsigned_char_array();
  
  conduit::Node converted_zbuf;
  depth_node["values"].to_unsigned_char_array(converted_zbuf);
  conduit::DataArray<unsigned char> zbuff_arr = converted_zbuf.as_unsigned_char_array();
  
  if( pixels_arr.number_of_elements() != img_width*img_height*bflow_comp::ImageData::sNUM_CHANNELS ||
      zbuff_arr.number_of_elements() != img_width*img_height )
  {
    std::cerr << "BFlowCompose: pixels_arr num elems = " << pixels_arr.number_of_elements() << std::endl;
    std::cerr << "BFlowCompose: zbuff_arr num elems = " << zbuff_arr.number_of_elements() << std::endl;
    ASCENT_ERROR("BabelFlow comp extract pixel array or zbuf array element count problem");
  }
  
  bflow_comp::ImageData input_img;
  
  input_img.image = new unsigned char[img_width*img_height*bflow_comp::ImageData::sNUM_CHANNELS];
  input_img.zbuf = new unsigned char[img_width*img_height];
  memcpy(input_img.image, pixels_arr.data_ptr(), pixels_arr.number_of_elements());
  memcpy(input_img.zbuf, zbuff_arr.data_ptr(), zbuff_arr.number_of_elements());
  input_img.bounds = new uint32_t[4];
  input_img.rend_bounds = new uint32_t[4];
  input_img.bounds[0] = input_img.rend_bounds[0] = 0;
  input_img.bounds[1] = input_img.rend_bounds[1] = img_width - 1;
  input_img.bounds[2] = input_img.rend_bounds[2] = 0;
  input_img.bounds[3] = input_img.rend_bounds[3] = img_height - 1;

  MPI_Comm mpi_comm;
  int my_rank = 0, n_ranks = 1;
#ifdef ASCENT_MPI_ENABLED
  mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
  MPI_Comm_rank(mpi_comm, &my_rank);
  MPI_Comm_size(mpi_comm, &n_ranks);
#endif
  
  int64_t fanin = p["fanin"].as_int64();
  CompositingType compositing_flag = CompositingType(p["compositing"].as_int64());
  std::string image_name = p["image_name"].as_string();
  
#ifdef BFLOW_COMP_DEBUG
  {
    std::stringstream img_name;
    img_name << "img_data_" << my_rank << "_" 
             << input_img.rend_bounds[0] << "_"
             << input_img.rend_bounds[1] << "_"
             << input_img.rend_bounds[2] << "_"
             << input_img.rend_bounds[3] << ".png";
    input_img.writeImage(img_name.str().c_str(), input_img.rend_bounds);
  }
#endif

  switch (compositing_flag)
  {
    case CompositingType::REDUCE:
      {
        int32_t n_blocks[3] = {1, 1, 1};
        bflow_comp::BabelCompReduce red_graph(input_img,
                                              image_name,
                                              my_rank, 
                                              n_ranks,
                                              fanin,
                                              mpi_comm,
                                              n_blocks);

        red_graph.Initialize();
        red_graph.Execute();
      }
      break;
    case CompositingType::BINSWAP:
      {
        bflow_comp::BabelCompBinswap binswap_graph(input_img,
                                                   image_name,
                                                   my_rank, 
                                                   n_ranks,
                                                   fanin,
                                                   mpi_comm);
        binswap_graph.Initialize();
        binswap_graph.Execute();
      }
      break;
    case CompositingType::RADIX_K:
      {
        std::vector<uint32_t> radix_v(1);
        radix_v[0] = n_ranks;
        if(p.has_path("radices"))
        {
          conduit::DataArray<int64_t> radices_arr = p["radices"].as_int64_array();
          radix_v.resize(radices_arr.number_of_elements());
          for (uint32_t i = 0; i < radix_v.size(); ++i) radix_v[i] = (uint32_t)radices_arr[i];
        }
        bflow_comp::BabelCompRadixK radixk_graph(input_img,
                                                 image_name,
                                                 my_rank, 
                                                 n_ranks,
                                                 fanin,
                                                 mpi_comm,
                                                 radix_v);
        radixk_graph.Initialize();
        radixk_graph.Execute();
      }
      break;
  }

  input_img.delBuffers();
}

