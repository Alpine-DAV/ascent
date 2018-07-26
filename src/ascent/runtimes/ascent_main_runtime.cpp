//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
#ifdef ASCENT_MPI_ENABLED
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
    ResetInfo();
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
    int rank = 0;
#if ASCENT_MPI_ENABLED
    if(!options.has_child("mpi_comm") ||
       !options["mpi_comm"].dtype().is_integer())
    {
        ASCENT_ERROR("Missing Ascent::Open options missing MPI communicator (mpi_comm)");
    }
    
    flow::Workspace::set_default_mpi_comm(options["mpi_comm"].to_int());
    vtkh::SetMPICommHandle(options["mpi_comm"].to_int());
    MPI_Comm comm = MPI_Comm_f2c(options["mpi_comm"].to_int());
    MPI_Comm_rank(comm,&rank);
#ifdef VTKM_CUDA
    //
    //  If we are using cuda, figure out how many devices we have and
    //  assign a GPU based on rank.
    //
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0 && device_count <= 256)
    {
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
#else  // non mpi version
    if(options.has_child("mpi_comm"))
    {
        // error if user tries to use mpi, but the version of ascent
        // the loaded version is actually the non-mpi version.
        ASCENT_ERROR("Ascent::Open MPI communicator (mpi_comm) passed to "
                     "non-mpi ascent.\n Are you linking and loading the "
                     "correct version of ascent?");
    }
    
#endif

    m_runtime_options = options;
    
    // standard flow filters
    flow::filters::register_builtin();
    // filters for ascent flow runtime.
    runtime::filters::register_builtin();
    
    if(options.has_path("web/stream") && 
       options["web/stream"].as_string() == "true" &&
       rank == 0)
    {
        
        if(options.has_path("web/document_root"))
        {
            m_web_interface.SetDocumentRoot(options["web/document_root"].as_string());
        }

        m_web_interface.Enable();
    }

    Node msg;
    this->Info(msg["info"]);
    ascent::about(msg["about"]);
    m_web_interface.PushMessage(msg);
    
}


//-----------------------------------------------------------------------------
void
AscentRuntime::Info(conduit::Node &out)
{
    out.set(m_info);
}

//-----------------------------------------------------------------------------
void
AscentRuntime::ResetInfo()
{
    m_info.reset();
    m_info["runtime/type"] = "ascent";
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
    conduit::Node multi_dom;
    blueprint::mesh::to_multi_domain(data, multi_dom);
    m_data.set_external(multi_dom);
    EnsureDomainIds();
}

//-----------------------------------------------------------------------------
void
AscentRuntime::EnsureDomainIds()
{
    // if no domain ids were provided add them now
    int num_domains = 0;
    bool has_ids = true;
    bool no_ids = true;
  
    // get the number of domains and check for id consistency
    num_domains = m_data.number_of_children();

    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = m_data.child(i);
      if(dom.has_path("state/domain_id"))
      {
        no_ids = false; 
      }
      else
      {
        has_ids = false;
      }
    }
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    int comm_id =flow::Workspace::default_mpi_comm();

    MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
    MPI_Comm_rank(mpi_comm,&rank);

    int comm_size = vtkh::GetMPISize();
    int *has_ids_array = new int[comm_size];
    int *no_ids_array = new int[comm_size];
    int boolean = has_ids ? 1 : 0; 

    MPI_Allgather(&boolean, 1, MPI_INT, has_ids_array, 1, MPI_INT, mpi_comm);
    boolean = no_ids ? 1 : 0; 
    MPI_Allgather(&boolean, 1, MPI_INT, no_ids_array, 1, MPI_INT, mpi_comm);

    bool global_has_ids = true;
    bool global_no_ids = false;
    for(int i = 0; i < comm_size; ++i)
    {
      if(has_ids_array[i] == 0)
      {
        global_has_ids = false;
      }
      if(no_ids_array[i] == 1)
      {
        global_no_ids = true;
      }
    }
    has_ids = global_has_ids;
    no_ids = global_no_ids;
    delete[] has_ids_array;
    delete[] no_ids_array;
#endif
      
    bool consistent_ids = (has_ids || no_ids);
     
    if(!consistent_ids)
    {
      ASCENT_ERROR("Inconsistent domain ids: all domains must either have an id "
                  <<"or all domains do not have an id");
    }

    int domain_offset = 0;
#ifdef ASCENT_MPI_ENABLED
    int *domains_per_rank = new int[comm_size];
    MPI_Allgather(&num_domains, 1, MPI_INT, domains_per_rank, 1, MPI_INT, mpi_comm);
    for(int i = 0; i < rank; ++i)
    {
      domain_offset += domains_per_rank[i];
    }
    delete[] domains_per_rank;  
#endif
    for(int i = 0; i < num_domains; ++i)
    {
      conduit::Node &dom = m_data.child(i);      

      int domain_id = domain_offset;
      if(!dom.has_path("state/domain_id"))
      {
         dom["state/domain_id"] = domain_offset + i;
      }
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
AscentRuntime::ConvertPipelineToFlow(const conduit::Node &pipeline,
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

      std::string type = filter["type"].as_string();
      
      bool has_params = filter.has_path("params");
      bool needs_params = true; 

      if(type == "contour")
      {
        filter_name = "vtkh_marchingcubes";
      }
      else if(type == "threshold")
      {
        filter_name = "vtkh_threshold";
      }
      else if(type == "clip")
      {
        filter_name = "vtkh_clip";
      }
      else if(type == "clip_with_field")
      {
        filter_name = "vtkh_clip_with_field";
      }
      else if(type == "iso_volume")
      {
        filter_name = "vtkh_iso_volume";
      }
      else if(type == "slice")
      {
        filter_name = "vtkh_slice";
      }
      else if(type == "3slice")
      {
        filter_name = "vtkh_3slice";
        needs_params = false;
      }
      else if(type == "NoOp")
      {
        filter_name = "vtkh_no_op";
      }
      else
      {
        ASCENT_ERROR("Unrecognized filter "<<filter["type"].as_string());
      }

      if(!has_params && needs_params)
      {
        filter.print();
        ASCENT_ERROR("Filter "<<type<<" must  declare a 'params'");
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
                  <<" over writing original");
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
    ConvertPipelineToFlow(pipe, names[i]);
  }
}

//-----------------------------------------------------------------------------
void 
AscentRuntime::ConvertExtractToFlow(const conduit::Node &extract,
                                    const std::string extract_name)
{
  std::string filter_name; 

  conduit::Node params;
  if(extract.has_path("params")) params = extract["params"];

  if(!extract.has_path("type"))
  {
    ASCENT_ERROR("Extract must have a 'type'");
  }
 
  if(extract["type"].as_string() == "adios")
  {
    filter_name = "adios";
  }
  else if(extract["type"].as_string() == "relay")
  {
    filter_name = "relay_io_save";
    // set the default protocol
    if(!params.has_path("protocol"))
    {
      params["protocol"] = "blueprint/mesh/hdf5";
    }
  }
  else if(extract["type"].as_string() == "python")
  {
    filter_name = "python_script";
    
    // customize the names of the script integration funcs
    params["interface/input"] = "ascent_data";
    params["interface/set_output"] = "ascent_set_output";

#ifdef ASCENT_MPI_ENABLED
    // for MPI case, inspect args, if script is passed via file,
    // read contents on root and broadcast to other tasks
    int comm_id =flow::Workspace::default_mpi_comm();
    MPI_Comm comm = MPI_Comm_f2c(comm_id);
    int rank = relay::mpi::rank(comm);
    MPI_Comm_rank(comm,&rank);
    
     if(params.has_path("file"))
     {
       Node n_py_src;
       // read script only on rank 0 
       if(rank == 0)
       {
         ostringstream py_src; 
         std::string script_fname = params["file"].as_string();
         ifstream ifs(script_fname.c_str());
         
         py_src << "# script from: " << script_fname << std::endl;
         copy(istreambuf_iterator<char>(ifs),
              istreambuf_iterator<char>(),
              ostreambuf_iterator<char>(py_src));
         n_py_src.set(py_src.str());
       }

       relay::mpi::broadcast_using_schema(n_py_src,0,comm);
       
       if(!n_py_src.dtype().is_string())
       {
         ASCENT_ERROR("broadcast of python script source failed");
       }
       // replace file param with source that includes actual script
       params.remove("file");
       params["source"] = n_py_src;
     }

     // inject helper that provides the mpi comm handle
     ostringstream py_src_final;
     py_src_final << "# ascent mpi comm helper function" << std::endl
                  << "def ascent_mpi_comm_id():" << std::endl
                  << "    return " << comm_id << std::endl
                  << std::endl
                  << params["source"].as_string(); // now include user's script

     params["source"] = py_src_final.str();

#endif
  }
  else
  {
    ASCENT_ERROR("Unrecognized extract type "<<extract["type"].as_string());
  }
  if(w.graph().has_filter(extract_name))
  {
    ASCENT_ERROR("Cannot add extract filter, extract named" 
                 << " \"" << extract_name << "\""
                 << " already exists");
  }


  std::string ensure_name = "ensure_blueprint_" + extract_name;
  conduit::Node empty_params; 
  
  w.graph().add_filter("ensure_blueprint",
                       ensure_name,
                       empty_params);

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
  
  m_connections[ensure_name] = extract_source;
  m_connections[extract_name] = ensure_name;

}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void 
AscentRuntime::ConvertPlotToFlow(const conduit::Node &plot,
                                 const std::string plot_name)
{
  std::string filter_name = "create_plot";; 

  if(w.graph().has_filter(plot_name))
  {
    ASCENT_INFO("Duplicate plot name "<<plot_name
                <<" over writing original");
  }

  w.graph().add_filter(filter_name,
                       plot_name,
                       plot);

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
    ConvertPlotToFlow(plot, names[i]);
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
AscentRuntime::ConnectSource()
{
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
void 
AscentRuntime::ConnectGraphs()
{
  //connect plot + pipeline graphs
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

//-----------------------------------------------------------------------------
std::vector<std::string>
AscentRuntime::GetPipelines(const conduit::Node &plots)
{
  std::vector<std::string> pipelines;
  std::vector<std::string> names = plots.child_names(); 
  for(int i = 0; i < plots.number_of_children(); ++i)
  {
    conduit::Node plot = plots.child(i);
    std::string pipeline;
    if(plot.has_path("pipeline"))
    {
      pipeline = plot["pipeline"].as_string();
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

    // ------------ NEW -----------------
    w.graph().add_filter("create_scene",
                          "create_scene_" + names[i]);

    std::string exec_name = "exec_" + names[i];
    w.graph().add_filter("exec_scene",
                          exec_name);

    // connect the renders to the scene exec
    // on the second port
    w.graph().connect(renders_name,   // src
                      exec_name,      // dest
                      1);             // default port

    // ------------ NEW -----------------
    
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
    
    //
    // as we add plots we aggregate them into the scene.
    // each add_plot filter output the sene and we have to
    // connect that to the next plot. At the end, we connect
    // the final output to the ExecPlot filter that actually 
    // calls render() on the scene.
    //
    std::string prev_add_plot_name = "";
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
          w.graph().connect(union_bounds_names[p-2],  // src
                            union_bounds_name,        // dest
                            0);                       // default port

          w.graph().connect(union_domain_ids_names[p-2],  // src
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

      // connect the plot with the scene
      std::string add_name = "add_plot_" + plot_names[p];
      w.graph().add_filter("add_plot",
                            add_name);
    
      std::string src_scene = prev_add_plot_name;

      if(prev_add_plot_name == "")
      {
        src_scene = "create_scene_" + names[i];
      }
      prev_add_plot_name = add_name; 
      
      // connect the plot to add_plot
      w.graph().connect(plot_names[p], // src
                        add_name,      // dest
                        1);            // plot port
      
      // connect the scene to add_plot
      w.graph().connect(src_scene,     // src
                        add_name,      // dest
                        0);            // scene port
      
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
    
    // connect Exec Scene to the output of the last
    // add_plot and to the renders
    w.graph().connect(prev_add_plot_name, // src
                      exec_name,          // dest
                      0);                 // default port

    w.graph().connect(renders_name,       // src
                      exec_name,          // dest
                      1);                 // default port
  } // each scene
}

void
AscentRuntime::FindRenders(const conduit::Node &info, 
                           conduit::Node &out)
{
    out.reset();    
    NodeConstIterator itr = info["flow_graph/graph/filters"].children();
    
    while(itr.has_next())
    {
        const Node &curr_filter = itr.next();
        ASCENT_INFO(curr_filter.to_json());
        if(curr_filter.has_path("params/image_prefix"))
        {
            std::string img_path = curr_filter["params/image_prefix"].as_string() + ".png";
            out.append() = img_path;
        }
    }
    
}

//-----------------------------------------------------------------------------
void
AscentRuntime::Execute(const conduit::Node &actions)
{
    ResetInfo();
    // make sure we always have our source data
    ConnectSource();
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
          w.info(m_info["flow_graph"]);
          w.execute();
          w.registry().reset();
          
          Node msg;
          this->Info(msg["info"]);
          ascent::about(msg["about"]);
          m_web_interface.PushMessage(msg);
          Node renders;
          FindRenders(msg["info"],renders);
          m_web_interface.PushRenders(renders);
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



