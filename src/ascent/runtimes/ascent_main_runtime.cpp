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

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>

#ifdef VTKM_CUDA
#include <vtkm/cont/cuda/ChooseCudaDevice.h>
#endif
#endif
using namespace conduit;
using namespace std;


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{
// node that holds tree of reg'd filter types
Node AscentRuntime::s_reged_filter_types;

class InfoHandler
{
  public:
  static int m_rank;
  static void
  info_handler(const std::string &msg,
               const std::string &file,
               int line)
  {
    if(m_rank == 0)
    {
      std::cout << "[" << file
                << " : " << line  << "]"
                << "\n " << msg << std::endl;
    }
  }
};

int InfoHandler::m_rank = 0;

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
AscentRuntime::AscentRuntime()
:Runtime(),
 m_refinement_level(2), // default refinement level for high order meshes
 m_rank(0),
 m_ghost_field_name("ascent_ghosts")
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
#if ASCENT_MPI_ENABLED
    if(!options.has_child("mpi_comm") ||
       !options["mpi_comm"].dtype().is_integer())
    {
        ASCENT_ERROR("Missing Ascent::Open options missing MPI communicator (mpi_comm)");
    }

    flow::Workspace::set_default_mpi_comm(options["mpi_comm"].to_int());
#if defined(ASCENT_VTKM_ENABLED)
    vtkh::SetMPICommHandle(options["mpi_comm"].to_int());
#endif
    MPI_Comm comm = MPI_Comm_f2c(options["mpi_comm"].to_int());
    MPI_Comm_rank(comm,&m_rank);
    InfoHandler::m_rank = m_rank;
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
    // set a info handler so we only display messages on rank 0;
    conduit::utils::set_info_handler(InfoHandler::info_handler);
#ifdef VTKM_CUDA

    bool sel_cuda_device = true;

    if(options.has_path("cuda/init") &&
       options["cuda/init"].as_string() == "false")
    {
        sel_cuda_device = false;
    }
    //
    //  If we are using cuda, figure out how many devices we have and
    //  assign a GPU based on rank.
    //
    if(sel_cuda_device)
    {
#if defined(ASCENT_VTKM_ENABLED)
        int device_count = vtkh::CUDADeviceCount();
        int rank_device = m_rank % device_count;
        vtkh::SelectCUDADevice(rank_device);
#endif
    }
#endif


#ifdef ASCENT_MFEM_ENABLED
    if(options.has_path("refinement_level"))
    {
      m_refinement_level = options["refinement_level"].to_int32();
      if(m_refinement_level < 2)
      {
        ASCENT_ERROR("'refinement_level' must be greater than 1");
      }
    }
#endif

    m_runtime_options = options;


    if(options.has_path("ghost_field_name"))
    {
      m_ghost_field_name = options["ghost_field_name"].as_string();
    }

    // standard flow filters
    flow::filters::register_builtin();
    // filters for ascent flow runtime.
    runtime::filters::register_builtin();

    if(options.has_path("web/stream") &&
       options["web/stream"].as_string() == "true" &&
       m_rank == 0)
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
    m_info["registered_filter_types"] = registered_filter_types();
}


//-----------------------------------------------------------------------------
void
AscentRuntime::Cleanup()
{
    if(m_runtime_options.has_child("timings") &&
       m_runtime_options["timings"].as_string() == "enabled")
    {
        // save out timing info on close
        std::stringstream fname;
        fname << "ascent_filter_times";

#ifdef ASCENT_MPI_ENABLED
        fname << "_" << m_rank;
#endif
        fname << ".csv";
        std::ofstream ftimings;
        ftimings.open(fname.str());
        ftimings << w.timing_info();
        ftimings.close();
    }
}

//-----------------------------------------------------------------------------
void
AscentRuntime::Publish(const conduit::Node &data)
{
    // create our own tree, with all data zero copied.
    blueprint::mesh::to_multi_domain(data, m_data);
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


#ifdef ASCENT_MPI_ENABLED
    int comm_id = flow::Workspace::default_mpi_comm();

    MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);

    int comm_size = 1;
    MPI_Comm_size(mpi_comm, &comm_size);
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
    for(int i = 0; i < m_rank; ++i)
    {
      domain_offset += domains_per_rank[i];
    }
    delete[] domains_per_rank;
#endif
    for(int i = 0; i < num_domains; ++i)
    {
      conduit::Node &dom = m_data.child(i);

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
    static std::string end_filter = "vtkh_data";
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
    // conver high order MFEM meshes to low order
    conduit::Node low_params;
    w.graph().add_filter("ensure_low_order",
                         "low_order",
                         low_params);

    w.graph().connect("verify",
                      "low_order",
                      0);        // default port

    conduit::Node vtkh_params;
    vtkh_params["zero_copy"] = "true";


    w.graph().add_filter("ensure_vtkh",
                         "vtkh_data",
                         vtkh_params);

    w.graph().connect("low_order",
                      "vtkh_data",
                      0);        // default port
    //if(m_has_ghosts)
    //{
      const std::string strip_name = "strip_garbage_ghosts";
      // garbage zones have a value of 2
      conduit::Node threshold_params;
      threshold_params["field"] = m_ghost_field_name;
      threshold_params["min_value"] = 0;
      threshold_params["max_value"] = 1;

      w.graph().add_filter("vtkh_ghost_stripper",
                           strip_name,
                           threshold_params);

      w.graph().connect("vtkh_data",
                        strip_name,
                        0);        // default port

      end_filter = strip_name;
    //}

    return end_filter;
}
//-----------------------------------------------------------------------------
void
AscentRuntime::ConvertPipelineToFlow(const conduit::Node &pipeline,
                                     const std::string pipeline_name)
{
    std::string prev_name = CreateDefaultFilters();
    bool has_pipeline = false;
    std::string input_name;
    // check to see if there is a non-default input to this pipeline
    if(pipeline.has_path("pipeline"))
    {
      prev_name = pipeline["pipeline"].as_string();
      input_name = prev_name;

      has_pipeline = true;
    }

    const std::vector<std::string> &child_names = pipeline.child_names();

    for(int i = 0; i < pipeline.number_of_children(); ++i)
    {
      const std::string cname = child_names[i];
      if(cname == "pipeline")
      {
        // this is a child that is not a filter.
        // It specifices the input to the pipeline itself
        continue;
      }
      conduit::Node filter = pipeline.child(i);
      std::string filter_name;

      if(!filter.has_path("type"))
      {
        filter.print();
        ASCENT_ERROR("Filter must declare a 'type'");
      }

      std::string type = filter["type"].as_string();

      if(registered_filter_types()["transforms"].has_child(type))
      {
          filter_name = registered_filter_types()["transforms"][type].as_string();
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

      if((input_name == prev_name) && has_pipeline)
      {
        // connect this later so we can specify them in any order
        m_connections[name] = prev_name;
      }
      else
      {
        w.graph().connect(prev_name, // src
                          name,      // dest
                          0);        // default port
      }

      prev_name = name;
    }

    if(w.graph().has_filter(pipeline_name))
    {
      ASCENT_INFO("Duplicate pipeline name '"<<pipeline_name
                  <<"' this is usually the symptom of a larger problem."
                  <<" Locate the first error message to find the root cause");
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

  std::string extract_type = extract["type"].as_string();

  // current special case filter setup
  if(extract_type == "python")
  {
    filter_name = "python_script";

    // customize the names of the script integration module and funcs
    params["interface/module"] = "ascent_extract";
    params["interface/input"]  = "ascent_data";
    params["interface/set_output"] = "ascent_set_output";

#ifdef ASCENT_MPI_ENABLED
    // for MPI case, inspect args, if script is passed via file,
    // read contents on root and broadcast to other tasks
    int comm_id = flow::Workspace::default_mpi_comm();
    MPI_Comm comm = MPI_Comm_f2c(comm_id);

     if(params.has_path("file"))
     {
       Node n_py_src;
       // read script only on rank 0
       if(m_rank == 0)
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
                  // bind ascent_mpi_comm_id into the module
                  << "ascent_extract.ascent_mpi_comm_id = ascent_mpi_comm_id"
                  << std::endl
                  << params["source"].as_string(); // now include user's script

     params["source"] = py_src_final.str();

#endif
  }
  // generic extract support
  else if(registered_filter_types()["extracts"].has_child(extract_type))
  {
     filter_name = registered_filter_types()["extracts"][extract_type].as_string();
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

  // currently these are special cases.
  // TODO:
  bool special = false;
  if(extract_type == "xray" ||
     extract_type == "volume") special = true;

  std::string ensure_name = "ensure_blueprint_" + extract_name;
  conduit::Node empty_params;
  if(!special)
  {
    w.graph().add_filter("ensure_blueprint",
                         ensure_name,
                         empty_params);
  }

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
    //current special case
    if(special)
    {
      extract_source = "default";
    }


  }
  if(!special)
  {
    m_connections[ensure_name] = extract_source;
    m_connections[extract_name] = ensure_name;
  }
  else
  {
    m_connections[extract_name] = extract_source;
  }

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
    ASCENT_INFO("Duplicate plot name '"<<plot_name
                <<"' this is usually the symptom of a larger problem."
                <<" Locate the first error message to find the root cause");
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
    plot_source = CreateDefaultFilters();
  }


  // we need to make sure that ghost zones don't make it into rendering
  // so we will create new filters that attach to the pipeline outputs
  std::string strip_name = plot_source + "_strip_real_ghosts";
  if(!w.graph().has_filter(strip_name))
  {
    conduit::Node threshold_params;
    threshold_params["field"] = m_ghost_field_name;
    threshold_params["min_value"] = 0;
    threshold_params["max_value"] = 0;

    w.graph().add_filter("vtkh_ghost_stripper",
                         strip_name,
                         threshold_params);

    w.graph().connect(plot_source,
                      strip_name,
                      0);        // default port
  }

  plot_source = strip_name;

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

void
AscentRuntime::PopulateMetadata()
{
  // add global state meta data to the registry
  const int num_domains = m_data.number_of_children();
  int cycle = 0;
  float time = 0.f;

  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = m_data.child(i);
    if(dom.has_path("state/cycle"))
    {
      cycle = dom["state/cycle"].to_int32();
    }
    if(dom.has_path("state/time"))
    {
      time = dom["state/time"].to_float32();
    }
  }

  if(!w.registry().has_entry("metadata"))
  {
    conduit::Node *meta = new conduit::Node();
    w.registry().add<Node>("metadata", meta,1);
  }

  Node *meta = w.registry().fetch<Node>("metadata");
  (*meta)["cycle"] = cycle;
  (*meta)["time"] = time;
  (*meta)["refinement_level"] = m_refinement_level;

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
// This function is used to feed renders (domain ids and bounds)
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

    // we are always adding a ghost filter so append the name
    // so bounds and domain ids get the right input
    pipeline = pipeline + "_strip_real_ghosts";
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
      ASCENT_ERROR("Scene must have at least one plot: "<<scene.to_json());
    }

    // create the default render
    conduit::Node render_params;
    if(scene.has_path("renders"))
    {
      render_params["renders"] = scene["renders"];
    }

    if(scene.has_path("image_prefix"))
    {
      render_params["image_prefix"] = scene["image_prefix"].as_string();
    }
    else
    {
      std::string image_prefix = GetDefaultImagePrefix(names[i]);
      render_params["image_prefix"] = image_prefix;
    }

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
    int plot_count = scene["plots"].number_of_children();
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

    if(!w.registry().has_entry("image_list"))
    {
      return;
    }

    Node *image_list = w.registry().fetch<Node>("image_list");

    const int size = image_list->number_of_children();
    for(int i = 0; i < size; i++)
    {
      out.append() = image_list->child(i).as_string();
    }

    out.print();

    image_list->reset();

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

        if(action_name == "add_pipelines")
        {
          if(action.has_path("pipelines"))
          {
            CreatePipelines(action["pipelines"]);
          }
          else
          {
            ASCENT_ERROR("action 'add_pipelines' missing child 'pipelines'");
          }
        }

        if(action_name == "add_scenes")
        {
          if(action.has_path("scenes"))
          {
            CreateScenes(action["scenes"]);
          }
          else
          {
            ASCENT_ERROR("action 'add_scenes' missing child 'scenes'");
          }
        }

        if(action_name == "add_extracts")
        {
          if(action.has_path("extracts"))
          {
            CreateExtracts(action["extracts"]);
          }
          else
          {
            ASCENT_ERROR("action 'add_extracts' missing child 'extracts'");
          }
        }

        else if( action_name == "execute")
        {
          ConnectGraphs();
          PopulateMetadata(); // add metadata so filters can access it
          w.info(m_info["flow_graph"]);
          //w.print();
          //std::cout<<w.graph().to_dot();

#if defined(ASCENT_VTKM_ENABLED)
          // we have vtkm enabled so catch any errors that
          // come up here and forward them up as a conduit
          // error
          try
          {
            w.execute();
          }
          catch(vtkh::Error &e)
          {
            ASCENT_ERROR("Execution failed with: "<<e.what());
          }
#else
          w.execute();
#endif

          Node msg;
          this->Info(msg["info"]);
          ascent::about(msg["about"]);
          m_web_interface.PushMessage(msg);
          Node renders;
          FindRenders(msg["info"],renders);
          m_web_interface.PushRenders(renders);

          w.registry().reset();
        }
        else if( action_name == "reset")
        {
          w.reset();
        }
    }
}

void
AscentRuntime::DisplayError(const std::string &msg)
{
  if(m_rank == 0)
  {
    std::cerr<<msg;
  }
}
//-----------------------------------------------------------------------------
void
AscentRuntime::RegisterFilterType(const std::string  &role_path,
                                  const std::string &api_name,
                                  const std::string &filter_type_name)
{
    std::string path = role_path;
    if(path == "")
    {
        path = "filters";
    }

    std::string f_name = api_name;

    if(f_name == "")
    {
        f_name = filter_type_name;
    }

    if(s_reged_filter_types[path].has_child(f_name))
    {
        //ASCENT_ERROR("Filter " << f_name << " already registered at " << path);
    }
    else
    {
        s_reged_filter_types[path][f_name] = filter_type_name;
    }
}





//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



