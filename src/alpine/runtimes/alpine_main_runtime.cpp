//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: alpine_main_runtime.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_main_runtime.hpp"

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
#include <alpine_runtime_filters.hpp>

using namespace conduit;
using namespace std;


//-----------------------------------------------------------------------------
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
AlpineRuntime::AlpineRuntime()
:Runtime()
{
    flow::filters::register_builtin();
}

//-----------------------------------------------------------------------------
AlpineRuntime::~AlpineRuntime()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main runtime interface methods called by the alpine interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
AlpineRuntime::Initialize(const conduit::Node &options)
{
#if PARALLEL
    if(!options.has_child("mpi_comm") ||
       !options["mpi_comm"].dtype().is_integer())
    {
        ALPINE_ERROR("Missing Alpine::Open options missing MPI communicator (mpi_comm)");
    }
    
    flow::Workspace::set_default_mpi_comm(options["mpi_comm"].as_int());
    
#endif

    m_runtime_options = options;
    
    // standard flow filters
    flow::filters::register_builtin();
    // filters for alpine flow runtime.
    runtime::filters::register_builtin();
}


//-----------------------------------------------------------------------------
void
AlpineRuntime::Cleanup()
{

}

//-----------------------------------------------------------------------------
void
AlpineRuntime::Publish(const conduit::Node &data)
{
    // create our own tree, with all data zero copied.
    m_data.set_external(data);
    
    // note: if the reg entry for data was already added
    // the set_external updates everything,
    // we don't need to remove and re-add.
    if(!w.registry().has_entry("_alpine_input_data"))
    {
        w.registry().add<Node>("_alpine_input_data",
                               &m_data);
    }

    if(!w.graph().has_filter(":source"))
    {
       Node p;
       p["entry"] = "_alpine_input_data";
       w.graph().add_filter("registry_source",":source",p);
    }
}

//-----------------------------------------------------------------------------
std::string 
AlpineRuntime::CreateDefaultFilters()
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
    
    w.graph().connect(":source",
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
AlpineRuntime::ConvertToFlowGraph(const conduit::Node &pipeline,
                                  const std::string pipeline_name)
{
    std::string prev_name = CreateDefaultFilters(); 

    for(int i = 0; i < pipeline.number_of_children(); ++i)
    {
      conduit::Node filter = pipeline.child(i);
      std::string name;
      std::string filter_name;

      if(!filter.has_path("type"))
      {
        filter.print();
        ALPINE_ERROR("Filter must declare a 'type'");
      }

      if(!filter.has_path("params"))
      {
        filter.print();
        ALPINE_ERROR("Filter must declare a 'params'");
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
        ALPINE_ERROR("Unrecognized filter "<<filter["type"].as_string());
      }

      name = pipeline_name + "_" + filter_name;
      

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
      ALPINE_INFO("Duplicate pipeline name "<<pipeline_name
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
AlpineRuntime::CreatePipelines(const conduit::Node &pipelines)
{
  std::vector<std::string> names = pipelines.child_names(); 
  for(int i = 0; i < pipelines.number_of_children(); ++i)
  {
    
    std::cout<<"Pipeline name "<<names[i]<<"\n";
    conduit::Node pipe = pipelines.child(i);
    ConvertToFlowGraph(pipe, names[i]);
  }
}

//-----------------------------------------------------------------------------
void 
AlpineRuntime::ConvertPlotToFlow(const conduit::Node &plot,
                                 const std::string plot_name)
{
  std::string filter_name; 

  if(!plot.has_path("type"))
  {
    ALPINE_ERROR("Plot must have a 'type'");
  }
 
  if(plot["type"].as_string() == "pseudocolor")
  {
    filter_name = "vtkh_raytracer";

  }
  else if(plot["type"].as_string() == "volume")
  {
    filter_name = "vtkh_volume";
  }
  else
  {
    ALPINE_ERROR("Unrecognized plot type "<<plot["plot_type"].as_string());
  }
 
  if(w.graph().has_filter(plot_name))
  {
    ALPINE_INFO("Duplicate plot name "<<plot_name
                <<" original is being overwritted");
  }

  w.graph().add_filter(filter_name,
                       plot_name,           
                       plot["params"]);

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
    plot_source = ":source";
  }

  m_connections[plot_name] = plot_source;

}
//-----------------------------------------------------------------------------
void 
AlpineRuntime::CreatePlots(const conduit::Node &plots)
{
  plots.print();
  std::vector<std::string> names = plots.child_names(); 
  for(int i = 0; i < plots.number_of_children(); ++i)
  {
    std::cout<<"plot name "<<names[i]<<"\n";
    conduit::Node plot = plots.child(i);
    ConvertPlotToFlow(plot, names[i]);
  }
}
//-----------------------------------------------------------------------------
void 
AlpineRuntime::ConnectGraphs()
{
  //create plot + pipine graphs
  m_connections.print(); 
  std::vector<std::string> names = m_connections.child_names(); 
  for (int i = 0; i < m_connections.number_of_children(); ++i)
  { 
    std::string pipeline = m_connections[names[i]].as_string(); 
    if(!w.graph().has_filter(pipeline))
    {
      ALPINE_ERROR(names[i]<<"' references unknown pipeline: "<<pipeline);
    }

    w.graph().connect(pipeline, // src
                      names[i], // dest
                      0);       // default port
  }
}
//-----------------------------------------------------------------------------
void
AlpineRuntime::Execute(const conduit::Node &actions)
{
    actions.print(); 
    // Loop over the actions
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        string action_name = action["action"].as_string();

        ALPINE_INFO("Executing " << action_name);
        
        if(action_name == "add_pipelines")
        {
          CreatePipelines(action["pipelines"]);
        }

        if(action_name == "add_plots")
        {
          CreatePlots(action["plots"]);
        }
        
        ////////
        // TODO: imp "alpine pipeline"
        ////////
        
        // implement actions
/*
        if(action_name == "add_filter")
        {
            if(action.has_child("params"))
            {
                w.graph().add_filter(action["type_name"].as_string(),
                                     action["name"].as_string(),
                                     action["params"]);
            }
            else
            {
                w.graph().add_filter(action["type_name"].as_string(),
                                     action["name"].as_string());
            }
        }
        else if( action_name == "add_filters")
        {
            w.graph().add_filters(action["filters"]);
        }
        else if( action_name == "connect")
        {
            if(action.has_child("port"))
            {
                w.graph().connect(action["src"].as_string(),
                                  action["dest"].as_string(),
                                  action["port"].as_string());
            }
            else
            {
                // if no port, assume input 0
                w.graph().connect(action["src"].as_string(),
                                  action["dest"].as_string(),
                                  0);
            }
        }
        else if( action_name == "add_connections")
        {
            w.graph().add_connections(action["connections"]);
        }
        else if( action_name == "add_graph")
        {
            w.graph().add_graph(action["graph"]);
        }
        else if( action_name == "load_graph")
        {
            w.graph().load(action["path"].as_string());
        }
        else if( action_name == "save_graph")
        {
            w.graph().save(action["path"].as_string());
        }
        else if( action_name == "execute")
        {
            w.execute();
            w.registry().reset();
        }
        else if( action_name == "reset")
        {
            w.reset();
        }
       */    
    }
    ConnectGraphs();
    w.execute();
    w.registry().reset();
    //ExecutePlots();
}






//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



