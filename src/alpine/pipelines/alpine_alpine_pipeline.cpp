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
/// file: alpine_alpine_pipeline.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_alpine_pipeline.hpp"

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

#include <alpine_flow.hpp>
#include <alpine_flow_pipeline_filters.hpp>

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
AlpinePipeline::AlpinePipeline()
:Pipeline()
{
    flow::filters::register_builtin();
}

//-----------------------------------------------------------------------------
AlpinePipeline::~AlpinePipeline()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main pipeline interface methods called by the alpine interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
AlpinePipeline::Initialize(const conduit::Node &options)
{
#if PARALLEL
    if(!options.has_child("mpi_comm") ||
       !options["mpi_comm"].dtype().is_integer())
    {
        ALPINE_ERROR("Missing Alpine::Open options missing MPI communicator (mpi_comm)");
    }
    
    flow::Workspace::set_default_mpi_comm(options["mpi_comm"].as_int());
    
#endif

    m_pipeline_options = options;
    
    // standard flow filters
    flow::filters::register_builtin();
    // filters for apline flow pipeline.
    pipeline::flow::filters::register_builtin();
}


//-----------------------------------------------------------------------------
void
AlpinePipeline::Cleanup()
{

}

//-----------------------------------------------------------------------------
void
AlpinePipeline::Publish(const conduit::Node &data)
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
conduit::Node ConvertToFlowGraph(const conduit::Node &pipeline)
{
   
    Node graph;
    // always add verify
    graph["filters/verify/type_name"] = "blueprint_verify";
    graph["filters/verify/params/protocol"] = "mesh";
    // TODO: recognize who owns what filters and convert data sets
    graph["filters/vtkh_data/type_name"] = "ensure_vtkh";
   
    int conn_id = 0;
    graph["connections"].append();
    graph["connections"][conn_id]["src"] = ":source";
    graph["connections"][conn_id]["dest"] = "verify";

    conn_id++;

    graph["connections"].append();
    graph["connections"][conn_id]["src"] = "verify";
    graph["connections"][conn_id]["dest"] = "vtkh_data";

    conn_id++;

    graph["connections"].append();
    graph["connections"][conn_id]["src"] = "vtkh_data";
    graph["connections"][conn_id]["dest"] = "";

    conn_id++;

    conduit::Node filters = pipeline["filters"];
    std::string prev_name = "ensure";
    for(int i = 0; i < filters.number_of_children(); ++i)
    {
      conduit::Node filter = filters.child(i);
      std::string name;
      if(filter["filter_type"].as_string() == "contour")
      {
        name = "vtkh_contour";
        graph["filters/vtkh_contour/type_name"]  = "vtkh_marchingcubes";
        graph["filters/vtkh_contour/params/"] = filter["params"];

      }
      else if(filter["filter_type"].as_string() == "threshold")
      {
        name = "vtkh_thresh";
        graph["filters/vtkh_thresh/type_name"]  = "vtkh_threshold";
        graph["filters/vtkh_thresh/params"] = filter["params"]; 
      }
      else if(filter["filter_type"].as_string() == "clip")
      {
        name = "vtkh_clip";
        graph["filters/vtkh_clip/type_name"]  = "vtkh_clip";
        graph["filters/vtkh_clip/params/"] = filter["params"];
      }
      else
      {
        ALPINE_ERROR("Unrecognized filter "<<filter["filter_type"].as_string());
      }

      graph["connections"].append();
      graph["connections"][conn_id-1]["dest"] = name; 
      graph["connections"][conn_id]["src"] = name; 
      graph["connections"][conn_id]["dest"] = ""; 
      conn_id++;
      prev_name = name;
    }
  return graph;
}

//-----------------------------------------------------------------------------
void 
AlpinePipeline::CreatePipelines(const conduit::Node &pipelines)
{
  //pipelines.print();
  std::vector<std::string> names = pipelines.child_names(); 
  for(int i = 0; i < pipelines.number_of_children(); ++i)
  {
    
    std::cout<<"Pipeline name "<<names[i]<<"\n";
    conduit::Node pipe = pipelines.child(i);
    conduit::Node graph = ConvertToFlowGraph(pipe);
    if(m_flow_pipelines.has_path(names[i]))
    {
      ALPINE_ERROR("Duplicate pipeline name "<<names[i]);
    }
    m_flow_pipelines[names[i]] = graph;
  }
}

//-----------------------------------------------------------------------------
conduit::Node ConvertPlotToFlow(const conduit::Node &plot)
{
   
  Node graph;
  
  std::string name;
  if(!plot.has_path("plot_type"))
  {
    //plot.print();
    ALPINE_ERROR("Plot must have a 'plot_type'");
  }

  if(plot["plot_type"].as_string() == "pseudocolor")
  {
    name = "vtkh_raytracer";
    graph["filters/vtkh_raytracer/type_name"]  = "vtkh_raytracer";
    graph["filters/vtkh_raytracer/params/"] = plot["params"];

  }
  else if(plot["plot_type"].as_string() == "volume")
  {
    name = "vtkh_volume";
    graph["filters/vtkh_volume/type_name"]  = "vtkh_volume";
    graph["filters/vtkh_volume/params"] = plot["params"]; 
  }
  else
  {
    ALPINE_ERROR("Unrecognized plot type "<<plot["plot_type"].as_string());
  }
 
  if(plot.has_path("pipeline"))
  {
    graph["pipeline"] = plot["pipeline"];
  }

  //graph["connections"].append();
  //graph["connections"][0]["src"] = ""; 
  //graph["connections"][0]["dest"] = name; 
  //graph.print();

  return graph;
}
//-----------------------------------------------------------------------------
void 
AlpinePipeline::CreatePlots(const conduit::Node &plots)
{
  //plots.print();
  std::vector<std::string> names = plots.child_names(); 
  for(int i = 0; i < plots.number_of_children(); ++i)
  {
    
    //std::cout<<"plot name "<<names[i]<<"\n";
    conduit::Node plot = plots.child(i);
    conduit::Node graph = ConvertPlotToFlow(plot);
    if(m_plots.has_path(names[i]))
    {
      ALPINE_ERROR("Duplicate plot name "<<names[i]);
    }
    m_plots[names[i]] = graph;
  }
}
//-----------------------------------------------------------------------------
void 
AlpinePipeline::MergeGraphs()
{
  m_flow_graphs.reset();
  //create plot + pipine graphs
  //m_plots.print(); 
  std::vector<std::string> names = m_plots.child_names(); 
  for (int i = 0; i < m_plots.number_of_children(); ++i)
  { 
    Node &plot = m_plots.child(i);
    if(!plot.has_path("pipeline"))
    {
      plot["connections"].append();
      plot["connections"][0]["src"] = ":source"; 
      plot["connections"][0]["dest"] = names[i]; 
      m_flow_graphs[names[i]] = plot;
      continue;
    }
    std::string pipeline = plot["pipeline"].as_string();
    if(!m_flow_pipelines.has_path(pipeline))
    {
      ALPINE_ERROR("Plot '"<<names[i]<<"' references unknown pipeline: "<<pipeline);
    }

    conduit::Node graph = m_flow_pipelines[pipeline]; 
    int conn_count = graph["connections"].number_of_children();
    //plot["filters"].print();
  
    //std::vector<std::string> filter_names = graph["filters"].child_names(); 
    
    graph["filters/"+names[i]] = plot["filters"][0];
    graph["connections"][conn_count-1]["dest"] = names[i];

    w.graph().add_graph(graph);
    //graph.print();
  }
}
//-----------------------------------------------------------------------------
void
AlpinePipeline::Execute(const conduit::Node &actions)
{
    // start fresh
    m_flow_pipelines.reset();
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
    MergeGraphs();
    w.execute();
    w.registry().reset();
    //ExecutePlots();
}






//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



