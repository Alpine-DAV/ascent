//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_rover_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_rover_filters.hpp"
#include "ascent_runtime_param_check.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_data_object.hpp>
#include <ascent_metadata.hpp>
#include <ascent_string_utils.hpp>
#include <ascent_runtime_utils.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <rover.hpp>
#include <rover/utils/rover_logging.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#include <ascent_runtime_blueprint_filters.hpp>
#include <ascent_runtime_relay_filters.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

using namespace std;
using namespace conduit;
using namespace flow;
using namespace rover;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
RoverXRay::RoverXRay()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RoverXRay::~RoverXRay()
{
// empty
}

//-----------------------------------------------------------------------------
void
RoverXRay::declare_interface(Node &i)
{
  i["type_name"] = "xray";
  i["port_names"].append() = "in";
  i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
RoverXRay::verify_params(const conduit::Node &params,
                               conduit::Node &info)
{
  // TODO: We want to be more rigorous about param checking at some point, so
  // that rover can safely assume its inputs are valid
  info.reset();

  // Early return if none of the rover params were passed
  if (!params.has_child("rover"))
  {
    info["errors"].append() = "Missing required string parameters: 'rover/absorption', 'rover/filename'";
    return false;
  }

  const conduit::Node &n_rover = params["rover"];
  bool res = true;

  //
  // Required rover parameters
  //

  if (!n_rover.has_child("absorption"))
  {
    info["errors"].append() = "Missing required string parameter 'rover/absorption'";
    res = false;
  }
  else if (!n_rover["absorption"].dtype().is_string())
  {
    info["errors"].append() = "Expected string parameter 'rover/absorption' is not a string";
    res = false;
  }

  const std::string absorption = n_rover["absorption"].as_string();
  if (absorption.empty())
  {
    info["errors"].append() = "Expected string parameter 'rover/absorption' cannot be an empty string";
    res = false;
  }

  if (!n_rover.has_child("filename"))
  {
    info["errors"].append() = "Missing required string parameter 'rover/filename'";
    res = false;
  }
  else if (!n_rover["filename"].dtype().is_string())
  {
    info["errors"].append() = "Expected string parameter 'rover/filename' is not a string";
    res = false;
  }

  const std::string filename = n_rover["filename"].as_string();
  if (filename.empty())
  {
    info["errors"].append() = "Expected string parameter 'rover/filename' cannot be an empty string";
    res = false;
  }

  //
  // Optional rover parameters
  //

  if (n_rover.has_child("background_intensity"))
  {
    if (!n_rover["background_intensity"].dtype().is_number())
    {
      info["errors"].append() = "Optional numeric parameter 'rover/background_intensity' is not numeric";
      res = false;
    }
    else // (n_rover["background_intensity"].dtype().is_number()
    {
      const float64 background_intensity = n_rover["background_intensity"].to_float64();
      if (background_intensity < 0)
      {
        info["errors"].append() = "Optional numeric parameter 'rover/unit_scalar' must be positive";
        res = false;
      }
    }
  }

  const bool has_blueprint_output = n_rover.has_child("blueprint_output");
  if (has_blueprint_output)
  {
    if (!n_rover["blueprint_output"].dtype().is_string())
    {
      info["errors"].append() = "Optional bool string parameter 'rover/blueprint_output' is not a string";
      res = false;
    }
    
    const std::string blueprint_output = n_rover["blueprint_output"].as_string();
    if ("true" != blueprint_output && "false" != blueprint_output)
    {
      info["errors"].append() = "Optional bool string parameter 'rover/blueprint_output' must be 'true' or 'false'";
      res = false;
    }
  }

  const bool has_blueprint_protocol = n_rover.has_child("blueprint_protocol");
  if (has_blueprint_protocol)
  {
    if (!n_rover["blueprint_protocol"].dtype().is_string())
    {
      info["errors"].append() = "Optional string parameter 'rover/blueprint_protocol' is not a string";
      res = false;
    }
    
    const std::string blueprint_protocol = n_rover["blueprint_protocol"].as_string();
    if ("hdf5" != blueprint_protocol && "yaml" != blueprint_protocol && "json" != blueprint_protocol)
    {
      info["errors"].append() = "Optional string parameter 'rover/blueprint_protocol' must be 'hdf5' or 'yaml' or 'json'";
      res = false;
    }
  }
  
  // If either 'rover/blueprint_output' or 'rover/blueprint_protocol' are set, they must both be set
  if (has_blueprint_output && !has_blueprint_protocol)
  {
    info["errors"].append() = "Optional bool string parameter 'rover/blueprint_output' requires 'rover/blueprint_protocol' to also be set";
    res = false;
  }
  else if (!has_blueprint_output && has_blueprint_protocol)
  {
    info["errors"].append() = "Optional string parameter 'rover/blueprint_protocol' requires 'rover/blueprint_output' to also be set";
    res = false;
  }

  const bool has_bov_output = n_rover.has_child("bov_output");
  if (has_bov_output)
  {
    if (!n_rover["bov_output"].dtype().is_string())
    {
      info["errors"].append() = "Optional bool string parameter 'rover/bov_output' is not a string";
      res = false;
    }
    
    const std::string bov_output = n_rover["bov_output"].as_string();
    if ("true" != bov_output && "false" != bov_output)
    {
      info["errors"].append() = "Optional bool string parameter 'rover/bov_output' must be 'true' or 'false'";
      res = false;
    }
  }

  if (n_rover.has_child("divide_emis_by_absorb"))
  {
    if (!n_rover["divide_emis_by_absorb"].dtype().is_string())
    {
      info["errors"].append() = "Optional bool string parameter 'rover/divide_emis_by_absorb' is not a string";
      res = false;
    }
    else // (n_rover["divide_emis_by_absorb"].dtype().is_string())
    {
      const std::string divide_emis_by_absorb = n_rover["divide_emis_by_absorb"].as_string();
      if ("true" != divide_emis_by_absorb && "false" != divide_emis_by_absorb)
      {
        info["errors"].append() = "Optional bool string parameter 'rover/divide_emis_by_absorb' must be 'true' or 'false'";
        res = false;
      }
    }
  }

  if (n_rover.has_child("emission"))
  {
    if (!n_rover["emission"].dtype().is_string())
    {
      info["errors"].append() = "Optional string parameter 'rover/emission' is not a string";
      res = false;
    }

    const std::string emission = n_rover["emission"].as_string();
    if (emission.empty())
    {
      info["errors"].append() = "Optional string parameter 'rover/emission' cannot be an empty string";
      res = false;
    }
  }

  // TODO: Remove this once issue #1559 is fixed
  if (n_rover.has_child("enable_imaging_planes"))
  {
    if (!n_rover["enable_imaging_planes"].dtype().is_string())
    {
      info["errors"].append() = "Optional bool string parameter 'rover/enable_imaging_planes' is not a string";
      res = false;
    }

    const std::string enable_imaging_planes = n_rover["enable_imaging_planes"].as_string();
    if ("true" != enable_imaging_planes && "false" != enable_imaging_planes)
    {
      info["errors"].append() = "Optional bool string parameter 'rover/enable_imaging_planes' must be 'true' or 'false'";
      res = false;
    }
  }

  const bool has_height = n_rover.has_child("height");
  if (has_height)
  {
    if (!n_rover["height"].dtype().is_integer())
    {
      info["errors"].append() = "Optional integer parameter 'rover/height' is not an integer";
      res = false;
    }
    else // (n_rover["height"].dtype().is_integer())
    {
      const int64 height = n_rover["height"].to_int64();
      if (height <= 0)
      {
        info["errors"].append() = "Optional integer parameter 'rover/height' must be greater than 0";
        res = false;
      }
    }
  }

  const bool has_png_output = n_rover.has_child("png_output");
  if (has_png_output)
  {
    if (!n_rover["bov_output"].dtype().is_string())
    {
      info["errors"].append() = "Optional bool string parameter 'rover/png_output' is not a string";
      res = false;
    }
    
    const std::string png_output = n_rover["png_output"].as_string();
    if ("true" != png_output && "false" != png_output)
    {
      info["errors"].append() = "Optional bool string parameter 'rover/png_output' must be 'true' or 'false'";
      res = false;
    }
  }

  // This won't be necessary once rover becomes a filter
  if (!has_blueprint_output && !has_bov_output && !has_png_output)
  {
    info["errors"].append() = "Expected at least one output type. Options are 'rover/blueprint_output', 'rover/bov_output', or 'rover/png_output'";
    res = false;
  }

  if (n_rover.has_child("precision"))
  {
    if (!n_rover["precision"].dtype().is_string())
    {
      info["errors"].append() = "Optional string parameter 'rover/precision' is not a string";
      res = false;
    }
    else // (n_rover["precision"].dtype().is_string())
    {
      const std::string precision = n_rover["precision"].as_string();
      if ("single" != precision && "double" != precision)
      {
        info["errors"].append() = "Optional string parameter 'rover/precision' must be 'single' or 'double'";
        res = false;
      }
    }
  }

  const bool has_width = n_rover.has_child("width");
  if (has_width)
  {
    if (!n_rover["width"].dtype().is_integer())
    {
      info["errors"].append() = "Optional integer parameter 'rover/width' is not an integer";
      res = false;
    }
    else // (n_rover["width"].dtype().is_integer())
    {
      const int64 width = n_rover["width"].to_int64();
      if (width <= 0)
      {
        info["errors"].append() = "Optional integer parameter 'rover/width' must be greater than 0";
        res = false;
      }
    }
  }

  // If either 'rover/width' or 'rover/height' are set, they must both be set
  if (has_width && !has_height)
  {
    info["errors"].append() = "Optional integer parameter 'rover/width' requires 'rover/height' to also be set";
    res = false;
  }
  else if (!has_width && has_height)
  {
    info["errors"].append() = "Optional integer parameter 'rover/height' requires 'rover/width' to also be set";
    res = false;
  }

  if (n_rover.has_child("unit_scalar"))
  {
    if (!n_rover["unit_scalar"].dtype().is_number())
    {
      info["errors"].append() = "Optional numeric parameter 'rover/unit_scalar' is not numeric";
      res = false;
    }
    else // (n_rover["unit_scalar"].dtype().is_number()
    {
      const float64 unit_scalar = n_rover["unit_scalar"].to_float64();
      if (unit_scalar <= 0)
      {
        info["errors"].append() = "Optional numeric parameter 'rover/unit_scalar' must be greater than 0";
        res = false;
      }
    }
  }

  //
  // Optional image parameters
  //

  if (params.has_child("image_params"))
  {
    // If any 'image_params' parameters are set, they must all be set
    const conduit::Node &n_image = params["image_params"];

    if (!n_image.has_child("log_scale"))
    {
      info["errors"].append() = "Missing bool string parameter 'image_params/log_scale'";
      res = false;
    }
    else if (!n_image["log_scale"].dtype().is_string())
    {
      info["errors"].append() = "Optional bool string parameter 'image_params/log_scale' is not a string";
      res = false;
    }
    else // (n_image.has_child("log_scale") && n_image["log_scale"].dtype().is_string())
    {
      const std::string log_scale = n_image["log_scale"].as_string();
      if ("true" != log_scale && "false" != log_scale)
      {
        info["errors"].append() = "Optional bool string parameter 'image_params/log_scale' must be 'true' or 'false'";
        res = false;
      }
    }

    if (!n_image.has_child("min_value"))
    {
      info["errors"].append() = "Missing numeric parameter 'image_params/min_value'";
      res = false;
    }
    else if (!n_image["min_value"].dtype().is_number())
    {
      info["errors"].append() = "Expected numeric parameter 'image_params/min_value' is not numeric";
      res = false;
    }

    if (!n_image.has_child("max_value"))
    {
      info["errors"].append() = "Missing numeric parameter 'image_params/max_value'";
      res = false;
    }
    else if (!n_image["max_value"].dtype().is_number())
    {
      info["errors"].append() = "Expected numeric parameter 'image_params/max_value' is not numeric";
      res = false;
    }
  }

  //
  // Surprise check
  //
  
  const std::vector<std::string> valid_paths = {
    "camera/azimuth",
    "camera/elevation",
    "camera/far_plane",
    "camera/fov",
    "camera/look_at",
    "camera/near_plane",
    "camera/up",
    "camera/xpan",
    "camera/ypan",
    "camera/zoom",
    "image_params/log_scale",
    "image_params/max_value",
    "image_params/min_value",
    "rover/absorption",
    "rover/background_intensity",
    "rover/blueprint_output",
    "rover/blueprint_protocol",
    "rover/bov_output",
    "rover/divide_emis_by_absorb",
    "rover/emission",
    "rover/enable_imaging_planes", // TODO: Remove this once #1559 is fixed
    "rover/filename",
    "rover/height",
    "rover/png_output",
    "rover/precision",
    "rover/width",
    "rover/unit_scalar"
  };

  const std::string surprises = surprise_check(valid_paths, params);
  if ("" != surprises)
  {
    info["errors"].append() = surprises;
    res = false;
  }

  return res;
}

//-----------------------------------------------------------------------------
void
RoverXRay::execute()
{
  // MPI
  int mpi_comm_id = -1;
  int mpi_rank = 0;
#ifdef ASCENT_MPI_ENABLED
  mpi_comm_id = flow::Workspace::default_mpi_comm();
  rover::Logger::get_instance()->set_mpi_comm_id(mpi_comm_id);
  rover::DataLogger::GetInstance()->set_mpi_comm_id(mpi_comm_id);
  MPI_Comm_rank(MPI_Comm_f2c(mpi_comm_id), &mpi_rank);
#endif

  const conduit::Node &n_params = params();
  const conduit::Node &n_rover = n_params["rover"];

  if (!input(0).check_type<DataObject>())
  {
    ASCENT_ERROR("Rover input must be a data object");
  }

  DataObject *data_object = input<DataObject>(0);
  if (!data_object->is_valid())
  {
    ASCENT_ERROR("Rover input must be a valid data object");
  }

  std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

  // Validate that the 'absorption' field exists in the dataset
  const std::string absorption = n_rover["absorption"].as_string();
  if(!collection->has_field(absorption))
  {
    ASCENT_ERROR("The dataset does not have a field called '" << absorption << "'");
  }

  // Validate that the 'absorption' field has a topology
  const std::string topo_name = collection->field_topology(absorption);
  if (topo_name.empty())
  {
    ASCENT_ERROR("The dataset does not have a topology associated with the '" << absorption << "' field");
  }

  // Validate that the dataset is non-empty
  vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);
  // Only rank 0 will have the data at this point
  if (0 == mpi_rank && dataset.IsEmpty())
  {
    ASCENT_ERROR("The dataset does not have a topololgy associated with the '" << absorption << "' field");
  }

  // Initialize rover and configure its behavior with the input params
  Rover rover;
#ifdef ASCENT_MPI_ENABLED
  rover.set_mpi_comm_handle(mpi_comm_id);
#endif
  rover.update_time_and_cycle(Metadata::n_metadata);
  rover.update_settings(n_params);
  
  // Adding a dataset to rover resets the camera bounds to the dataset bounds,
  // but any camera params passed via the input params will take precedence.
  // It also instantiates a scheduler if one doesn't already exist.
  rover.add_dataset(dataset);

  // Calling execute initializes everything that rover needs based on the input params
  rover.execute();

  //
  // Outputs
  //

  const std::string filename = n_rover["filename"].as_string();

  if (n_rover.has_child("blueprint_output"))
  {
    const std::string blueprint_output = n_rover["blueprint_output"].as_string();
    if ("true" == blueprint_output)
    {
      conduit::Node multi_domain;
      conduit::Node &data = multi_domain.append();
      rover.to_blueprint(data);
  
      const std::string blueprint_filename = output_dir(expand_path_special_variables(
                                                filename,
                                                ".root",
                                                mpi_comm_id));
      const std::string blueprint_protocol = n_rover["blueprint_protocol"].as_string();
      const int num_files = -1;
      conduit::Node extra_opts;
      std::string result_path;
      mesh_blueprint_save(multi_domain,
                          blueprint_filename,
                          blueprint_protocol,
                          num_files,
                          extra_opts,
                          result_path);
    }
  }  

  if (n_rover.has_child("bov_output"))
  {
    const std::string bov_output = n_rover["bov_output"].as_string();
    if ("true" == bov_output)
    {
      const std::string bov_filename = output_dir(expand_path_special_variables(
                                                    filename,
                                                    ".bov",
                                                    mpi_comm_id));
      rover.save_bov(bov_filename);
    }
  }

  if (n_rover.has_child("png_output"))
  {
    const std::string png_output = n_rover["png_output"].as_string();
    if ("true" == png_output)
    {
      ASCENT_WARN("Rover's png output is currently broken\n");
      const std::string png_filename = output_dir(expand_path_special_variables(
                                                    filename,
                                                    ".png",
                                                    mpi_comm_id));
      rover.save_png(png_filename);
    }
  }
}

#if 0 // removing volume renderer
//-----------------------------------------------------------------------------
RoverVolume::RoverVolume()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RoverVolume::~RoverVolume()
{
// empty
}

//-----------------------------------------------------------------------------
void
RoverVolume::declare_interface(Node &i)
{
    i["type_name"]   = "rover_volume";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
RoverVolume::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("field") ||
       ! params["field"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'field'";
        res = false;
    }

    if(! params.has_child("filename") ||
       ! params["filename"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'filename'";
        res = false;
    }

    if( params.has_child("precision") &&
       ! params["precision"].dtype().is_string() )
    {
        info["errors"].append() = "Optional parameter 'precision' must be a string";
        std::string prec = params["precision"].as_string();
        if(prec != "single" || prec != "double")
        {
          info["errors"].append() = "Parameter 'precision' must be 'single' or 'double'";
        }
        res = false;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
RoverVolume::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("rover input must be a data object");
    }

    DataObject *data_object = input<DataObject>(0);
    if(!data_object->is_valid())
    {
      return;
    }
    std::shared_ptr<VTKHCollection> collection = data_object->as_vtkh_collection();

    std::string field_name = params()["field"].as_string();
    if(!collection->has_field(field_name))
    {
      ASCENT_ERROR("Unknown field '"<<field_name<<"'");
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);

    vtkmCamera camera;
    camera.ResetToBounds(dataset.GetGlobalBounds());

    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      parse_camera(n_camera, camera);
    }

    int width, height;
    parse_image_dims(params(), width, height);

    CameraGenerator generator(camera, width, height);

    Rover tracer;
    int mpi_comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
    mpi_comm_id =flow::Workspace::default_mpi_comm();
    tracer.set_mpi_comm_handle(mpi_comm_id);
#endif

    if(params().has_path("precision"))
    {
      std::string prec = params()["precision"].as_string();
      if(prec == "double")
      {
        tracer.set_tracer_precision64();
      }
    }

    //
    // Create some basic settings
    //
    RenderSettings settings;
    settings.m_primary_field = params()["field"].as_string();

    if(params().has_path("samples"))
    {
      settings.m_volume_settings.m_num_samples = params()["samples"].to_int32();
    }


    if(params().has_path("min_value"))
    {
      settings.m_volume_settings.m_scalar_range.Min = params()["min_value"].to_float32();
    }

    if(params().has_path("max_value"))
    {
      settings.m_volume_settings.m_scalar_range.Max = params()["max_value"].to_float32();
    }

    settings.m_render_mode = rover::volume;
    if(params().has_path("color_table"))
    {
      settings.m_color_table = parse_color_table(params()["color_table"]);
    }
    else
    {
      vtkmColorTable color_table("cool to warm");
      color_table.AddPointAlpha(0.0, .1);
      color_table.AddPointAlpha(0.5, .2);
      color_table.AddPointAlpha(1.0, .3);
      settings.m_color_table = color_table;
    }

    tracer.set_render_settings(settings);
    tracer.add_data_set(dataset);

    tracer.set_ray_generator(&generator);
    tracer.execute();

    std::string filename = params()["filename"].as_string();
    filename = output_dir(expand_path_special_variables(filename, ".png", mpi_comm_id));

    tracer.save_png(filename);
}
#endif

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





