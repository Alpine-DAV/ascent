//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
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
    for(int i = 0; i < dataset.GetNumberOfDomains(); ++i)
    {
      tracer.add_data_set(dataset.GetDomain(i));
    }

    tracer.set_ray_generator(&generator);
    tracer.execute();

    Node * meta = graph().workspace().registry().fetch<Node>("metadata");
    int cycle = -1;;
    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].as_int32();
    }

    std::string filename = params()["filename"].as_string();
    if(cycle != -1)
    {
      filename = expand_family_name(filename, cycle);
    }
    filename = output_dir(filename, graph());

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
      bov_filename = output_dir(bov_filename, graph());
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
    for(int i = 0; i < dataset.GetNumberOfDomains(); ++i)
    {
      tracer.add_data_set(dataset.GetDomain(i));
    }

    tracer.set_ray_generator(&generator);
    tracer.execute();

    int cycle = -1;;
    Node * meta = graph().workspace().registry().fetch<Node>("metadata");
    if(meta->has_path("cycle"))
    {
      cycle = (*meta)["cycle"].as_int32();
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
    filename = output_dir(filename, graph());

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





