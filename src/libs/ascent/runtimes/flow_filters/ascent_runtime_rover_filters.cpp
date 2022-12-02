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


using namespace conduit;
using namespace std;

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
    i["type_name"]   = "xray";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
RoverXRay::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("absorption") ||
       ! params["absorption"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'absorption'";
        res = false;
    }

    if(! params.has_child("filename") ||
       ! params["filename"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'filename'";
        res = false;
    }

    if( params.has_child("emission") &&
       ! params["emission"].dtype().is_string() )
    {
        info["errors"].append() = "Optional parameter 'emission' must be a string";
        res = false;
    }


    if(params.has_path("image_params"))
    {
      if( !params.has_path("image_params/log_scale") ||
         ! params["image_params/log_scale"].dtype().is_string() )
      {
          info["errors"].append() = "Missing required image parameter 'log_scale' must be a string";
          res = false;
      }

      if( !params.has_path("image_params/min_value") ||
         ! params["image_params/min_value"].dtype().is_number() )
      {
          info["errors"].append() = "Missing required image parameter 'min_value' must be a number";
          res = false;
      }

      if( !params.has_path("image_params/max_value") ||
         ! params["image_params/max_value"].dtype().is_number() )
      {
          info["errors"].append() = "Missing required image parameter 'max_value' must be a number";
          res = false;
      }
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
RoverXRay::execute()
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

    std::string field_name = params()["absorption"].as_string();
    if(!collection->has_field(field_name))
    {
      ASCENT_ERROR("Unknown field '"<<field_name<<"'");
    }

    std::string topo_name = collection->field_topology(field_name);

    vtkm::cont::PartitionedDataSet &dataset = collection->dataset_by_topology(topo_name);

    vtkmCamera camera;
    camera.ResetToBounds(vtkh::GetGlobalBounds(dataset));

    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      parse_camera(n_camera, camera);
    }

    int width, height;
    parse_image_dims(params(), width, height);

    CameraGenerator generator(camera, width, height);

    Rover tracer;
#ifdef ASCENT_MPI_ENABLED
    int comm_id = flow::Workspace::default_mpi_comm();
    tracer.set_mpi_comm_handle(comm_id);
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
    settings.m_primary_field = params()["absorption"].as_string();

    if(params().has_path("emission"))
    {
       settings.m_secondary_field = params()["emission"].as_string();
    }

    if(params().has_path("unit_scalar"))
    {
       settings.m_energy_settings.m_unit_scalar = params()["unit_scalar"].to_float64();
    }


    settings.m_render_mode = rover::energy;

    tracer.set_render_settings(settings);
    std::vector<vtkm::cont::DataSet> v_datasets = dataset.GetPartitions();
    for(int i = 0; i < dataset.GetNumberOfPartitions(); ++i)
    {
      tracer.add_data_set(v_datasets[i]);
    }

    tracer.set_ray_generator(&generator);
    tracer.execute();

    Node meta = Metadata::n_metadata;
    int cycle = -1;
    if(meta.has_path("cycle"))
    {
      cycle = meta["cycle"].as_int32();
    }

    std::string filename = params()["filename"].as_string();
    if(cycle != -1)
    {
      filename = expand_family_name(filename, cycle);
    }

    filename = output_dir(filename);

    if(params().has_path("blueprint"))
    {


      std::string protocol = params()["blueprint"].as_string();
      conduit::Node multi_domain;
      conduit::Node &dom = multi_domain.append();
      tracer.to_blueprint(dom);

      if(dom.has_path("coordsets"))
      {
        int cycle = -1;
        double time = -1.;

        if(Metadata::n_metadata.has_path("cycle"))
        {
          cycle = Metadata::n_metadata["cycle"].to_int32();
        }
        if(Metadata::n_metadata.has_path("time"))
        {
          time = Metadata::n_metadata["time"].to_float64();
        }

        if(cycle != -1)
        {
          dom["state/cycle"] = cycle;
        }

        if(time != -1.)
        {
          dom["state/time"] = time;
        }
      }

      std::string result_path;
      mesh_blueprint_save(multi_domain, filename, protocol, -1, result_path);
    }

    if(params().has_path("image_params"))
    {
      float min_value = params()["image_params/min_value"].to_float32();
      float max_value = params()["image_params/max_value"].to_float32();
      bool log_scale = params()["image_params/log_scale"].as_string() == "true";
      tracer.save_png(filename, min_value, max_value, log_scale);

    }
    else
    {
      tracer.save_png(filename);
    }

    if(params().has_path("bov_filename"))
    {
      std::string bov_filename = params()["bov_filename"].as_string();
      bov_filename = output_dir(bov_filename);
      if(cycle != -1)
      {
        tracer.save_bov(expand_family_name(bov_filename, cycle));
      }
      else
      {
        tracer.save_bov(expand_family_name(bov_filename));
      }
    }
    tracer.finalize();

}

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

    vtkm::cont::PartitionedDataSet &dataset = collection->dataset_by_topology(topo_name);

    vtkmCamera camera;
    camera.ResetToBounds(vtkh::GetGlobalBounds(dataset));

    if(params().has_path("camera"))
    {
      const conduit::Node &n_camera = params()["camera"];
      parse_camera(n_camera, camera);
    }

    int width, height;
    parse_image_dims(params(), width, height);

    CameraGenerator generator(camera, width, height);

    Rover tracer;
#ifdef ASCENT_MPI_ENABLED
    int comm_id =flow::Workspace::default_mpi_comm();
    tracer.set_mpi_comm_handle(comm_id);
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
    std::vector<vtkm::cont::DataSet> v_datasets = dataset.GetPartitions();
    for(int i = 0; i < dataset.GetNumberOfPartitions(); ++i)
    {
      tracer.add_data_set(v_datasets[i]);
    }

    tracer.set_ray_generator(&generator);
    tracer.execute();

    Node meta = Metadata::n_metadata;
    int cycle = -1;
    if(meta.has_path("cycle"))
    {
      cycle = meta["cycle"].as_int32();
    }

    std::string filename = params()["filename"].as_string();
    if(cycle != -1)
    {
      filename = expand_family_name(filename, cycle);
    }
    else
    {
      filename = expand_family_name(filename);
    }
    filename = output_dir(filename);

    tracer.save_png(filename);
    tracer.finalize();

}

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





