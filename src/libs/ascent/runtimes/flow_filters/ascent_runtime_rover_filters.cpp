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

#if defined(ASCENT_VISKORES_ENABLED)
#include <rover.hpp>
#include <rover/utils/rover_logging.hpp>
#include <vtkh/vtkh.hpp>
#include <vtkh/DataSet.hpp>
#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_viskores_parsing.hpp>
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

  // ----------- Define Param Schema -----------
  conduit::Node &param_schema = i["param_schema"];
  param_schema["type"] = "object";
  param_schema["additionalProperties"] = false;

  string_schema(param_schema["properties/condition"]);
  string_schema(param_schema["properties/callback"]);
  string_schema(param_schema["properties/actions_file"]);
  array_schema(param_schema["properties/actions_files"]);
  array_schema(param_schema["properties/actions"]);

  // --- Rover ---
  conduit::Node &rover_schema = param_schema["properties/rover"];
  string_schema(rover_schema["properties/absorption"], 1);
  string_schema(rover_schema["properties/filename"], 1);
  number_schema(rover_schema["properties/background_intensity"], false, 0);
  bool_schema(rover_schema["properties/divide_emis_by_absorb"]);
  string_schema(rover_schema["properties/emission"], 1);
  bool_schema(rover_schema["properties/enable_rays_mesh"]);
  integer_schema(rover_schema["properties/height"], false, 0, std::numeric_limits<int>::max(), 0);
  integer_schema(rover_schema["properties/width"], false, 0, std::numeric_limits<int>::max(), 0);
  string_enum_schema(rover_schema["properties/output_type"], {"hdf5", "yaml", "json", "png", "bov"});
  string_enum_schema(rover_schema["properties/precision"], {"single", "double"});
  number_schema(rover_schema["properties/unit_scalar"], false, 0, std::numeric_limits<int>::max(), 0);

  rover_schema["constraints/dependencies/height"].append() = "width";
  rover_schema["constraints/dependencies/width"].append() = "height";

  rover_schema["required"].append() = "absorption";
  rover_schema["required"].append() = "filename";

  // --- Image ---
  conduit::Node &image_schema = param_schema["properties/image_params"];
  bool_schema(image_schema["properties/log_scale"]);
  number_schema(image_schema["properties/min_value"]);
  number_schema(image_schema["properties/max_value"]);

  // --- Camera ---
  conduit::Node &camera_schema = param_schema["properties/camera"];
  ignore_schema(camera_schema["properties/azimuth"]);
  ignore_schema(camera_schema["properties/elevation"]);
  ignore_schema(camera_schema["properties/far_plane"]);
  ignore_schema(camera_schema["properties/near_plane"]);
  ignore_schema(camera_schema["properties/fov"]);
  ignore_schema(camera_schema["properties/look_at"]);
  ignore_schema(camera_schema["properties/position"]);
  ignore_schema(camera_schema["properties/up"]);
  ignore_schema(camera_schema["properties/xpan"]);
  ignore_schema(camera_schema["properties/ypan"]);
  ignore_schema(camera_schema["properties/zoom"]);

  param_schema["required"].append() = "rover";
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
  // rover::Logger::get_instance()->set_mpi_comm_id(mpi_comm_id);
  // rover::DataLogger::GetInstance()->set_mpi_comm_id(mpi_comm_id);
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
  // It also instantiates one scheduler per MPI rank if they don't already exist.
  rover.add_dataset(dataset);

  // Calling execute initializes everything that rover needs based on the input params
  rover.execute();

  //
  // Outputs
  //

  // The default output is blueprint using the hdf5 protocol
  std::string output_type = "hdf5";
  if (n_rover.has_child("output_type"))
  {
    output_type = n_rover["output_type"].as_string();
  }

  const std::string filename = n_rover["filename"].as_string();

  if ("hdf5" == output_type || "yaml" == output_type || "json" == output_type)
  {
    conduit::Node multi_domain;
    conduit::Node &data = multi_domain.append();

    if (0 == mpi_rank)
    {
      rover.to_blueprint(data);
    }

    ASCENT_ANNOTATE_MARK_BEGIN("rover filter save blueprint");

    const std::string blueprint_filename = output_dir(expand_path_special_variables(
                                                      filename,
                                                      ".root",
                                                      mpi_comm_id));
    const int num_files = -1;
    conduit::Node extra_opts;
    std::string result_path;
    mesh_blueprint_save(multi_domain,
                        blueprint_filename,
                        output_type,
                        num_files,
                        extra_opts,
                        result_path);
    ASCENT_ANNOTATE_MARK_END("rover filter save blueprint");
  }  
  else if ("bov" == output_type)
  {
    const std::string bov_filename = output_dir(expand_path_special_variables(
                                                filename,
                                                ".bov",
                                                mpi_comm_id));
    rover.save_bov(bov_filename);
  }
  else if ("png" == output_type)
  {
    ASCENT_WARN("Rover's png output is currently broken\n");
    const std::string png_filename = output_dir(expand_path_special_variables(
                                                filename,
                                                ".png",
                                                mpi_comm_id));
    rover.save_png(png_filename);
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

    // ----------- Define Param Schema -----------
    conduit::Node &param_schema = i["param_schema"];
    param_schema["type"] = "object";
    param_schema["additionalProperties"] = false;

    string_schema(param_schema["properties/field"]);
    string_schema(param_schema["properties/filename"]);
    string_enum_schema(param_schema["properties/precision"], {"single", "double"});

    param_schema["required"].append() = "field";
    param_schema["required"].append() = "filename";
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

    viskoresCamera camera;
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
      viskoresColorTable color_table("cool to warm");
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





