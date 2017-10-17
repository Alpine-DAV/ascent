//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
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
/// file: ascent_main_runtime.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_main_runtime.hpp"

// standard lib includes
#include <string.h>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit_blueprint.hpp>

// mpi related includes
#ifdef PARALLEL
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

#include <flow.hpp>
#include <ascent_runtime_filters.hpp>

#include <vtkh/vtkh.hpp>

#ifdef VTKM_CUDA
#include <vtkm/cont/cuda/ChooseCudaDevice.h>
#endif

using namespace conduit;
using namespace std;


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
AscentRuntime::AscentRuntime()
:Runtime()
{
    flow::filters::register_builtin();
}

//-----------------------------------------------------------------------------
AscentRuntime::~AscentRuntime()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main runtime interface methods called by the ascent interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
AscentRuntime::Initialize(const conduit::Node &options)
{
#if PARALLEL
    if(!options.has_child("mpi_comm") ||
       !options["mpi_comm"].dtype().is_integer())
    {
        ASCENT_ERROR("Missing Ascent::Open options missing MPI communicator (mpi_comm)");
    }
    
    flow::Workspace::set_default_mpi_comm(options["mpi_comm"].as_int());
   
    MPI_Comm comm = MPI_Comm_f2c(options["mpi_comm"].as_int());
    vtkh::SetMPIComm(comm);
#ifdef VTKM_CUDA
    //
    //  If we are using cuda, figure out how many devices we have and
    //  assign a GPU based on rank.
    //
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0 && device_count <= 256)
    {
        int rank;  
        MPI_Comm_rank(comm,&rank);
        int rank_device = rank % device_count;
        err = cudaSetDevice(rank_device);
        if(err != cudaSuccess)
        {
            ASCENT_ERROR("Failed to set GPU " 
                           <<rank_device
                           <<" out of "<<device_count
                           <<" GPUs. Make sure there"
                           <<" are an equal amount of"
                           <<" MPI ranks/gpus per node.");
        }
        else
        {

            char proc_name[100];
            int length=0;
            MPI_Get_processor_name(proc_name, &length);

        }
        cuda_device  = rank_device;
    }
    else
    {
        ASCENT_ERROR("VTKm GPUs is enabled but none found");
    }
#endif
#endif

    m_runtime_options = options;
    
    // standard flow filters
    flow::filters::register_builtin();
    // filters for ascent flow runtime.
    runtime::filters::register_builtin();
}


//-----------------------------------------------------------------------------
void
AscentRuntime::Cleanup()
{

}

//-----------------------------------------------------------------------------
void
AscentRuntime::Publish(const conduit::Node &data)
{
    // create our own tree, with all data zero copied.
    m_data.set_external(data);
    
    // note: if the reg entry for data was already added
    // the set_external updates everything,
    // we don't need to remove and re-add.
    if(!w.registry().has_entry("_ascent_input_data"))
    {
        w.registry().add<Node>("_ascent_input_data",
                               &m_data);
    }

    if(!w.graph().has_filter("source"))
    {
       Node p;
       p["entry"] = "_ascent_input_data";
       w.graph().add_filter("registry_source","source",p);
    }
}

//-----------------------------------------------------------------------------
std::string 
AscentRuntime::CreateDefaultFilters()
{
    const std::string end_filter = "vtkh_data";
    if(w.graph().has_filter(end_filter))
    {
      return end_filter;
    }
    // 
    // Here we are creating the default set of filters that all
    // pipelines will connect to. It verifies the mesh and 
    // ensures we have a vtkh data set going forward.
    //
    conduit::Node params;
    params["protocol"] = "mesh";

    w.graph().add_filter("blueprint_verify", // registered filter name
                         "verify",           // "unique" filter name
                         params);
    
    w.graph().connect("source",
                      "verify",
                      0);        // default port

    w.graph().add_filter("ensure_vtkh",
                         "vtkh_data");

    w.graph().connect("verify",
                      "vtkh_data",
                      0);        // default port

    return end_filter;
}
//-----------------------------------------------------------------------------
void 
AscentRuntime::ConvertToFlowGraph(const conduit::Node &pipeline,
                                  const std::string pipeline_name)
{
    std::string prev_name = CreateDefaultFilters(); 

    for(int i = 0; i < pipeline.number_of_children(); ++i)
    {
      conduit::Node filter = pipeline.child(i);
      std::string filter_name;

      if(!filter.has_path("type"))
      {
        filter.print();
        ASCENT_ERROR("Filter must declare a 'type'");
      }

      if(!filter.has_path("params"))
      {
        filter.print();
        ASCENT_ERROR("Filter must declare a 'params'");
      }

      if(filter["type"].as_string() == "contour")
      {
        filter_name = "vtkh_marchingcubes";

      }
      else if(filter["type"].as_string() == "threshold")
      {
        filter_name = "vtkh_threshold";
      }
      else if(filter["type"].as_string() == "clip")
      {
        filter_name = "vtkh_clip";
      }
      else
      {
        ASCENT_ERROR("Unrecognized filter "<<filter["type"].as_string());
      }
     
      // create a unique name for the filter
      std::stringstream ss;
      ss<<pipeline_name<<"_"<<i<<"_"<<filter_name;
      std::string name = ss.str(); 
      
      w.graph().add_filter(filter_name,
                           name,           
                           filter["params"]);

      w.graph().connect(prev_name, // src
                        name,      // dest
                        0);        // default port
      prev_name = name;
    }
  
    if(w.graph().has_filter(pipeline_name))
    {
      ASCENT_INFO("Duplicate pipeline name "<<pipeline_name
                  <<" original is being overwritted");
    }
    // create an alias passthrough filter so plots and extracts
    // can connect to the end result by pipeline name
    w.graph().add_filter("alias",
                         pipeline_name);

    w.graph().connect(prev_name,     // src
                      pipeline_name, // dest
                      0);            // default port
}

//-----------------------------------------------------------------------------
void 
AscentRuntime::CreatePipelines(const conduit::Node &pipelines)
{
  std::vector<std::string> names = pipelines.child_names(); 
  for(int i = 0; i < pipelines.number_of_children(); ++i)
  {
    conduit::Node pipe = pipelines.child(i);
    ConvertToFlowGraph(pipe, names[i]);
  }
}

//-----------------------------------------------------------------------------
void 
AscentRuntime::ConvertExtractToFlow(const conduit::Node &extract,
                                    const std::string extract_name)
{
  std::string filter_name; 

  if(!extract.has_path("type"))
  {
    ASCENT_ERROR("Extract must have a 'type'");
  }
 
  if(extract["type"].as_string() == "adios")
  {
    filter_name = "adios";

  }
  else
  {
    ASCENT_ERROR("Unrecognized extract type "<<extract["type"].as_string());
  }
 
  if(w.graph().has_filter(extract_name))
  {
    ASCENT_INFO("Duplicate extract name "<<extract_name
                <<" original is being overwritted");
  }

  conduit::Node params = extract["params"];

  w.graph().add_filter(filter_name,
                       extract_name,           
                       params);

  //
  // We can't connect the extract to the pipeline since
  // we want to allow users to specify actions any any order 
  //
  std::string extract_source;
  if(extract.has_path("pipeline"))
  {
    extract_source = extract["pipeline"].as_string();;
  }
  else
  {
    // this is the blueprint mesh 
    extract_source = "source";
  }
  m_connections[extract_name] = extract_source;

}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void 
AscentRuntime::ConvertPlotToFlow(const conduit::Node &plot,
                                 const std::string plot_name,
                                 bool composite)
{
  std::string filter_name; 

  if(!plot.has_path("type"))
  {
    ASCENT_ERROR("Plot must have a 'type'");
  }
 
  if(plot["type"].as_string() == "pseudocolor")
  {
    filter_name = "vtkh_raytracer";

  }
  else if(plot["type"].as_string() == "volume")
  {
    filter_name = "vtkh_volumetracer";

  }
  else
  {
    ASCENT_ERROR("Unrecognized plot type "<<plot["type"].as_string());
  }
 
  if(w.graph().has_filter(plot_name))
  {
    ASCENT_INFO("Duplicate plot name "<<plot_name
                <<" original is being overwritted");
  }
  //
  // Plots are set to composite by default
  // and only the last plot should perform
  // compositing
  //
  conduit::Node params = plot["params"];
  if(!composite)
  {
    params["composite"] = "false";
  }
  w.graph().add_filter(filter_name,
                       plot_name,           
                       params);

  //
  // We can't connect the plot to the pipeline since
  // we want to allow users to specify actions any any order 
  //
  std::string plot_source;
  if(plot.has_path("pipeline"))
  {
    plot_source = plot["pipeline"].as_string();;
  }
  else
  {
    // default pipeline: directly connect to published data
    plot_source = "default";
  }
  m_connections[plot_name] = plot_source;

}
//-----------------------------------------------------------------------------
void 
AscentRuntime::CreatePlots(const conduit::Node &plots)
{
  std::vector<std::string> names = plots.child_names(); 
  for(int i = 0; i < plots.number_of_children(); ++i)
  {
    conduit::Node plot = plots.child(i);
    bool composite = i == plots.number_of_children() - 1;
    ConvertPlotToFlow(plot, names[i], composite);
  }
}
//-----------------------------------------------------------------------------
void 
AscentRuntime::CreateExtracts(const conduit::Node &extracts)
{
  std::vector<std::string> names = extracts.child_names(); 
  for(int i = 0; i < extracts.number_of_children(); ++i)
  {
    conduit::Node extract = extracts.child(i);
    ConvertExtractToFlow(extract, names[i]);
  }
}
//-----------------------------------------------------------------------------
void 
AscentRuntime::ConnectGraphs()
{
  //create plot + pipine graphs
  std::vector<std::string> names = m_connections.child_names(); 
  for (int i = 0; i < m_connections.number_of_children(); ++i)
  { 
    std::string pipeline = m_connections[names[i]].as_string(); 
    if(pipeline == "default")
    { 
      pipeline = CreateDefaultFilters(); 
    }
    else if(!w.graph().has_filter(pipeline))
    {
      ASCENT_ERROR(names[i]<<"' references unknown pipeline: "<<pipeline);
    }

    w.graph().connect(pipeline, // src
                      names[i], // dest
                      0);       // default port
  }
}

std::vector<std::string>
AscentRuntime::GetPipelines(const conduit::Node &plots)
{
  std::vector<std::string> pipelines;
  std::vector<std::string> names = plots.child_names(); 
  for(int i = 0; i < plots.number_of_children(); ++i)
  {
    conduit::Node plot = plots.child(i);
    std::string pipeline;
    if(plot.has_path("params/pipeline"))
    {
      pipeline = plot["params/pipeline"].as_string();
    }
    else
    {
      pipeline = CreateDefaultFilters(); 
    }
    pipelines.push_back(pipeline);
  }
  return pipelines;
}

std::string 
AscentRuntime::GetDefaultImagePrefix(const std::string scene)
{
  static conduit::Node image_counts;
  int count = 0;
  if(!image_counts.has_path(scene))
  {
    image_counts[scene] = count;
  }
  count = image_counts[scene].as_int32();
  image_counts[scene] = count + 1;
  
  std::stringstream ss;
  ss<<scene<<"_"<<count;
  return ss.str(); 
}

void
AscentRuntime::CreateScenes(const conduit::Node &scenes)
{

  std::vector<std::string> names = scenes.child_names(); 
  const int num_scenes = scenes.number_of_children();
  for(int i = 0; i < num_scenes; ++i)
  {
    conduit::Node scene = scenes.child(i);
    scene.print();
    if(!scene.has_path("plots"))
    {
      ASCENT_ERROR("Default scene not implemented");
    }

    // create the default render 
    conduit::Node render_params;
    if(scene.has_path("renders"))
    {
      render_params["renders"] = scene["renders"];
    } 
    int plot_count = scene["plots"].number_of_children();
    render_params["pipeline_count"] = plot_count;

    if(scene.has_path("image_prefix"))
    {
      render_params["image_prefix"] = scene["image_prefix"].as_string();;
    }
    else
    {
      std::string image_prefix = GetDefaultImagePrefix(names[i]); 
      render_params["image_prefix"] = image_prefix;
    }

    render_params["pipeline_count"] = plot_count;
    std::string renders_name = names[i] + "_renders";           
    
    w.graph().add_filter("default_render",
                          renders_name,
                          render_params);
    //
    // TODO: detect if there is a volume plot, rendering it last
    //
    std::vector<std::string> pipelines = GetPipelines(scene["plots"]); 
    std::vector<std::string> plot_names = scene["plots"].child_names();
    
    // create plots with scene name appended 
    conduit::Node appended_plots;
    for(int k = 0; k < plot_names.size(); ++k)
    {
      std::string p_name = plot_names[k] + "_" + names[i];
      appended_plots[p_name] = scene["plots/" + plot_names[k]];
    }

    plot_names = appended_plots.child_names();

    CreatePlots(appended_plots);

    std::vector<std::string> bounds_names;
    std::vector<std::string> union_bounds_names;
    std::vector<std::string> domain_ids_names;
    std::vector<std::string> union_domain_ids_names;
    
    for(int p = 0; p < plot_count; ++p)
    {
      //
      // To setup the rendering we need to setup a several filters.
      // A render, needs two inputs:
      //     1) a set of domain ids to create canvases
      //     2) the coordinate bounds of all the input pipelines
      //        to be able to create a defailt camera
      // Thus, we have to call bounds and domain id filters and
      // create unions of all inputs to feed to the render.
      //
      std::string bounds_name = plot_names[p] + "_bounds";
      conduit::Node empty; 
      w.graph().add_filter("vtkh_bounds",
                            bounds_name,
                            empty);

      w.graph().connect(pipelines[p], // src
                        bounds_name,  // dest
                        0);           // default port
      bounds_names.push_back(bounds_name);
  
      std::string domain_ids_name = plot_names[p] + "_domain_ids";
      w.graph().add_filter("vtkh_domain_ids",
                            domain_ids_name,
                            empty);

      w.graph().connect(pipelines[p],     // src
                        domain_ids_name,  // dest
                        0);               // default port
      domain_ids_names.push_back(domain_ids_name);
  
      //
      // we have more than one. Create union filters.
      //
      if(p > 0)
      {
        std::string union_bounds_name = plot_names[p] + "_union_bounds";
        w.graph().add_filter("vtkh_union_bounds",
                              union_bounds_name,
                              empty);
        union_bounds_names.push_back(union_bounds_name);

        std::string union_domain_ids_name = plot_names[p] + "_union_domain_ids";
        w.graph().add_filter("vtkh_union_domain_ids",
                              union_domain_ids_name,
                              empty);
        union_domain_ids_names.push_back(union_domain_ids_name);

        if(p == 1)
        {
          // first union just needs the output
          // of the first bounds
          w.graph().connect(bounds_names[p-1],  // src
                            union_bounds_name,  // dest
                            0);                 // default port

          w.graph().connect(domain_ids_names[p-1],  // src
                            union_domain_ids_name,  // dest
                            0);                     // default port
        }
        else
        {
          // all subsequent unions needs the bounds of 
          // the current plot and the output of the 
          // previous union
          //
          w.graph().connect(union_bounds_names[p-1],  // src
                            union_bounds_name,        // dest
                            0);                       // default port

          w.graph().connect(union_domain_ids_names[p-1],  // src
                            union_domain_ids_name,        // dest
                            0);                           // default port
        }

        w.graph().connect(bounds_name,        // src
                          union_bounds_name,  // dest
                          1);                 // default port

        w.graph().connect(domain_ids_name,        // src
                          union_domain_ids_name,  // dest
                          1);                     // default port
      }

      //
      // Connect the render to the plots
      //
      if(p == 0)
      {
        //
        // first plot connects to the render filter
        // on the second port
        w.graph().connect(renders_name,   // src
                          plot_names[p],  // dest
                          1);             // default port
      }
      else
      {
        //
        // Connect plot output to the next plot
        //
        w.graph().connect(plot_names[p-1],   // src
                          plot_names[p],     // dest
                          1);                // default port

      }
      
    }
  
    //
    // Connect the total bounds and domain ids
    // up to the render inputs
    //
    std::string bounds_output; 
    std::string domain_ids_output; 

    if(bounds_names.size() == 1)
    {
      bounds_output = bounds_names[0];
      domain_ids_output = domain_ids_names[0];
    }
    else
    {
      const size_t union_size = union_bounds_names.size();
      bounds_output = union_bounds_names[union_size-1];
      domain_ids_output = union_domain_ids_names[union_size-1];
    }

    w.graph().connect(bounds_output, // src
                      renders_name,  // dest
                      0);            // default port

    w.graph().connect(domain_ids_output, // src
                      renders_name,      // dest
                      1);                // default port
  }
}
//-----------------------------------------------------------------------------
void
AscentRuntime::Execute(const conduit::Node &actions)
{
    // Loop over the actions
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        string action_name = action["action"].as_string();

        ASCENT_INFO("Executing " << action_name);
        
        if(action_name == "add_pipelines")
        {
          CreatePipelines(action["pipelines"]);
        }

        if(action_name == "add_scenes")
        {
          CreateScenes(action["scenes"]);
        }
        
        if(action_name == "add_extracts")
        {
          CreateExtracts(action["extracts"]);
        }
        
        else if( action_name == "execute")
        {
          ConnectGraphs();
          ASCENT_INFO(w.graph().to_dot());
          w.execute();
          w.registry().reset();
        }
        else if( action_name == "reset")
        {
            w.reset();
        }
    }
}






//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



