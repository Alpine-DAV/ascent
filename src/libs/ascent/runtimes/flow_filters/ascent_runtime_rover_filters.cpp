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
#include <ray_generators/camera_generator.hpp>
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
  bool res = true;

  if (!params.has_path("rover/absorption") || !params["rover/absorption"].dtype().is_string())
  {
    info["errors"].append() = "Missing required string parameter 'rover/absorption'";
    res = false;
  }

  const std::string absorption = params["rover/absorption"].as_string();
  if ("" == absorption)
  {
    info["errors"].append() = "Required parameter 'rover/absorption' cannot be an empty string";
    res = false;
  }

  if (!params.has_path("rover/filename") || !params["rover/filename"].dtype().is_string())
  {
    info["errors"].append() = "Missing required string parameter 'rover/filename'";
    res = false;
  }

  if (params.has_path("rover/emission") && !params["rover/emission"].dtype().is_string())
  {
    info["errors"].append() = "Optional parameter 'rover/emission' must be a string";
    res = false;
  }

  const bool has_width = params.has_path("rover/width");
  const bool has_height = params.has_path("rover/height");

  if (has_width && has_height)
  {
    if (!params["rover/width"].dtype().is_integer())
    {
      info["errors"].append() = "Optional parameter 'rover/width' must be an integer";
      res = false;
    }
    else
    {
      const int64 width = params["rover/width"].to_int64();
      if (width <= 0)
      {
        info["errors"].append() = "Optional parameter 'rover/width' must be greater than 0";
        res = false;
      }
    }

    if (!params["rover/height"].dtype().is_integer())
    {
      info["errors"].append() = "Optional parameter 'rover/height' must be an integer";
      res = false;
    }
    else
    {
      const int64 height = params["rover/height"].to_int64();
      if (height <= 0)
      {
        info["errors"].append() = "Optional parameter 'rover/height' must be greater than 0";
        res = false;
      }
    }
  }
  else if (has_width && !has_height)
  {
    info["errors"].append() = "Optional parameter 'rover/width' requires 'rover/height' to also be set";
    res = false;
  }
  else if (!has_width && has_height)
  {
    info["errors"].append() = "Optional parameter 'rover/height' requires 'rover/width' to also be set";
    res = false;
  }

  if (params.has_child("image_params"))
  {
    if (!params.has_path("image_params/log_scale") ||
        !params["image_params/log_scale"].dtype().is_string())
    {
      info["errors"].append() = "Missing required image parameter 'log_scale' must be a string";
      res = false;
    }

    if (!params.has_path("image_params/min_value") ||
        !params["image_params/min_value"].dtype().is_number())
    {
      info["errors"].append() = "Missing required image parameter 'min_value' must be a number";
      res = false;
    }

    if (!params.has_path("image_params/max_value") ||
        !params["image_params/max_value"].dtype().is_number())
    {
      info["errors"].append() = "Missing required image parameter 'max_value' must be a number";
      res = false;
    }
  }

  if (params.has_path("rover/precision"))
  {
    if (!params["rover/precision"].dtype().is_string())
    {
      info["errors"].append() = "Optional parameter 'rover/precision' must be a string";
      info["errors"].append() = "Optional parameter 'rover/precision' must be 'single' or 'double'";
      res = false;
    }
    else
    {
      const std::string precision = params["rover/precision"].as_string();
      if (precision != "single" && precision != "double")
      {
        info["errors"].append() = "Optional parameter 'rover/precision' must be 'single' or 'double'";
        res = false;
      }
    }
  }

  if (params.has_path("rover/blueprint"))
  {
    const std::string protocol = params["rover/blueprint"].as_string();
    if (protocol != "hdf5" && protocol != "yaml" && protocol != "json")
    {
      info["errors"].append() = "Optional parameter 'rover/blueprint' must be 'hdf5' or 'yaml' or 'json'";
      res = false;
    }
  }

  return res;
}

//-----------------------------------------------------------------------------
void
RoverXRay::execute()
{
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
  const std::string absorption = params()["rover/absorption"].as_string();
  if(!collection->has_field(absorption))
  {
    ASCENT_ERROR("Absorption field name '" << absorption << "' is not in the dataset");
  }

  // Fetch the dataset associated with the 'absorption' field
  std::string topo_name = collection->field_topology(absorption);
  // TODO: Validate topo_name
  vtkh::DataSet &dataset = collection->dataset_by_topology(topo_name);
  // TODO: Validate dataset

  // Initialize rover and configure its behavior with the input params
  Rover rover;
  rover.update_metadata(Metadata::n_metadata);
  rover.update_settings(params());

  int mpi_comm_id = -1;
#ifdef ASCENT_MPI_ENABLED
  mpi_comm_id = flow::Workspace::default_mpi_comm();
  rover::Logger::get_instance()->set_mpi_comm_id(mpi_comm_id);
  rover::DataLogger::GetInstance()->set_mpi_comm_id(mpi_comm_id);
  rover.set_mpi_comm_handle(mpi_comm_id);
#endif
  
  // Adding a dataset to rover resets the camera bounds to the dataset bounds,
  // but any camera params passed via the input params will take precedence.
  // It also instantiates a scheduler if one doesn't already exist.
  rover.add_dataset(dataset);
  // Calling execute initializes everything that rover needs based on the input params
  rover.execute();

  if (params().has_path("rover/blueprint"))
  {
    conduit::Node multi_domain;
    conduit::Node &data = multi_domain.append();
    rover.to_blueprint(data);

    std::string filename = params()["rover/filename"].as_string();
    filename = output_dir(expand_path_special_variables(filename, ".root", mpi_comm_id));
    const std::string protocol = params()["rover/blueprint"].as_string();
    const int num_files = -1;
    conduit::Node extra_opts;
    std::string result_path;
    mesh_blueprint_save(multi_domain,
                        filename,
                        protocol,
                        num_files,
                        extra_opts,
                        result_path);
  }

  // TODO: I don't think we want to always save a .png unconditionally,
  // so maybe params could be reworked to request the exact types of output
  // the user wants rover to produce
  std::string png_filename = params()["rover/filename"].as_string();
  png_filename = output_dir(expand_path_special_variables(png_filename, ".png", mpi_comm_id));
  rover.save_png(png_filename);

  // TODO: We don't check if rover/bov_filename is valid in verify_params
  if (params().has_path("rover/bov_filename"))
  {
    std::string bov_filename = params()["rover/bov_filename"].as_string();
    bov_filename = output_dir(bov_filename);
    rover.save_bov(expand_path_special_variables(bov_filename, ".bov", mpi_comm_id));
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





