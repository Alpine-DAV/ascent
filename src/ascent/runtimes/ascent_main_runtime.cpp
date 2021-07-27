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
#include <algorithm>

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
#include <ascent_actions_utils.hpp>
#include <ascent_metadata.hpp>
#include <ascent_runtime_filters.hpp>
#include <ascent_expression_eval.hpp>
#include <expressions/ascent_blueprint_architect.hpp>
#include <ascent_transmogrifier.hpp>
#include <ascent_data_object.hpp>

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkm/cont/Error.h>
#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>

#ifdef VTKM_CUDA
#include <vtkm/cont/cuda/ChooseCudaDevice.h>
#endif
#endif

#if defined(ASCENT_DRAY_ENABLED)
#include <dray/dray.hpp>
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
 m_default_output_dir("."),
 m_session_name("ascent_session"),
 m_field_filtering(false)
{
    m_ghost_fields.append() = "ascent_ghosts";
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
#if defined(ASCENT_DRAY_ENABLED)
    dray::dray::mpi_comm(options["mpi_comm"].to_int());
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
      Transmogrifier::m_refinement_level = m_refinement_level;
      if(m_refinement_level < 1)
      {
        ASCENT_ERROR("'refinement_level' must be greater than 0");
      }
    }
#endif
    if(options.has_path("default_dir"))
    {
      std::string dir = options["default_dir"].as_string();

      if(!directory_exists(dir))
      {
        ASCENT_INFO("'default_dir' '"<<dir<<"' does not exist."
                    <<" Output dir will default to the cwd.");
      }
      m_default_output_dir = dir;
    }

    m_runtime_options = options;

    if(options.has_path("ghost_field_name"))
    {
      if(options["ghost_field_name"].dtype().is_string())
      {
        m_ghost_fields.reset();

        std::string ghost_name = options["ghost_field_name"].as_string();
        m_ghost_fields.append() = ghost_name;
      }
      else if(options["ghost_field_name"].dtype().is_list())
      {
        const int num_children = options["ghost_field_name"].number_of_children();
        for(int i = 0; i < num_children; ++i)
        {
          const conduit::Node &child = options["ghost_field_name"].child(i);
          if(!child.dtype().is_string())
          {
            ASCENT_ERROR("ghost_field_name list child is not a string");
          }
        }
      }
      else
      {
        ASCENT_ERROR("ghost_field_name is not a string or a list");
      }
    }

    // standard flow filters
    flow::filters::register_builtin();
    // filters for ascent flow runtime.
    runtime::filters::register_builtin();
    // filters for expression evaluation
    runtime::expressions::register_builtin();

    if(options.has_path("session_name"))
    {
      m_session_name = options["session_name"].as_string();
    }

    runtime::expressions::ExpressionEval::load_cache(m_default_output_dir,
                                                     m_session_name);

    if(options.has_path("web/stream") &&
       options["web/stream"].as_string() == "true" &&
       m_rank == 0)
    {
#ifdef ASCENT_WEBSERVER_ENABLED
        if(options.has_path("web/document_root"))
        {
            m_web_interface.SetDocumentRoot(options["web/document_root"].as_string());
        }

        m_web_interface.Enable();
#else
        ASCENT_ERROR("Ascent was not built with web support,"
                     "but options[\"web/stream\"] == \"true\"");
#endif
    }

    if(options.has_path("field_filtering"))
    {
      if(options["field_filtering"].as_string() == "true")
      {
        m_field_filtering = true;
      }
    }

    Node msg;
    ascent::about(msg["about"]);
    msg["options"] = options;
    this->Info(msg["info"]);
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
       m_runtime_options["timings"].as_string() == "true")
    {
        // save out timing info on close
        std::stringstream fname;
        fname << "ascent_filter_times";

#ifdef ASCENT_MPI_ENABLED
        fname << "_" << m_rank;
#endif
        fname << ".csv";
        std::ofstream ftimings;
        std::string file_name = fname.str();
        file_name = conduit::utils::join_file_path(m_default_output_dir,file_name);
        ftimings.open(file_name, std::ofstream::out | std::ofstream::app);
        ftimings << w.timing_info();
        ftimings.close();
    }
}

//-----------------------------------------------------------------------------
void
AscentRuntime::Publish(const conduit::Node &data)
{
    blueprint::mesh::to_multi_domain(data, m_source);
    EnsureDomainIds();
    // filter out default ghost name and
    // check if user provided ghost names are actually there
    VerifyGhosts();
    // if nestsets are present, agument current ghost fields
    // for zones masked by finer levels. If no ghosts are present
    // we create them
    PaintNestsets();
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
    num_domains = m_source.number_of_children();
    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = m_source.child(i);
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
      conduit::Node &dom = m_source.child(i);

      if(!dom.has_path("state/domain_id"))
      {
         dom["state/domain_id"] = domain_offset + i;
      }
    }
}

//-----------------------------------------------------------------------------
conduit::Node
AscentRuntime::CreateDefaultFilters()
{
    static std::string queries_endpoint = "default_queries_endpoint";
    static std::string endpoint = "default_filters_endpoint";

    conduit::Node endpoints;
    endpoints["filters"] = endpoint;
    endpoints["queries"] = queries_endpoint;

    if(w.graph().has_filter(endpoint))
    {
      return endpoints;
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

    std::string prev_filter = "verify";

#if defined(ASCENT_VTKM_ENABLED)
    // we can have multiple ghost fields
    std::vector<std::string> ghost_fields;
    const int num_children = m_ghost_fields.number_of_children();
    for(int i = 0; i < num_children; ++i)
    {
      ghost_fields.push_back(m_ghost_fields.child(i).as_string());
    }

    std::string first_stripper;
    const int num_ghosts = ghost_fields.size();
    for(int i = 0; i < num_ghosts; ++i)
    {
      std::string filter_name = "strip_garbage_" + ghost_fields[i];

      if(i == 0)
      {
        first_stripper = filter_name;
      }
      // garbage zones have a value of 2
      conduit::Node threshold_params;
      threshold_params["field"] = ghost_fields[i];
      threshold_params["min_value"] = 0;
      threshold_params["max_value"] = 1;

      w.graph().add_filter("vtkh_ghost_stripper",
                           filter_name,
                           threshold_params);

      w.graph().connect(prev_filter,
                        filter_name,
                        0);        // default port

      prev_filter = filter_name;
    }
#endif

    // we are creating a series of endpoints to enforce and
    // order of execution. Pipelines using expressions might
    // need the results of a query, so make them execute first
    // create an alias passthrough
    w.graph().add_filter("alias",
                         queries_endpoint);

    w.graph().connect(prev_filter,      // src
                      queries_endpoint, // dest
                      0);               // default port

    w.graph().add_filter("dependent_alias",
                         endpoint);

    w.graph().connect(queries_endpoint, // src
                      endpoint,         // dest
                      0);               // default port

    return endpoints;
}
//-----------------------------------------------------------------------------
void
AscentRuntime::ConvertPipelineToFlow(const conduit::Node &pipeline,
                                     const std::string pipeline_name)
{
    std::string prev_name = CreateDefaultFilters()["filters"].as_string();
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
        // It specifies the input to the pipeline itself
        continue;
      }

      if(cname == "type")
      {
        // this is common error to put a filter directly
        // under pipelines
        ASCENT_ERROR("Detected 'type' with a pipeline definition "<<
                     "("<<pipeline_name<<"). This occurs when a "
                     "filter type is mistakenly placed outside "
                     "a filter");

      }

      conduit::Node filter = pipeline.child(i);
      std::string filter_name;

      if(!filter.has_path("type"))
      {
        filter.print();
        ASCENT_ERROR("Filter must declare a 'type'");
      }

      std::string type = filter["type"].as_string();

      // support pipelines that specify "exa" style filters
      if(type.find("exa") == (size_t)0 &&
         type.size() > (size_t)3)
      {
          type = type.substr(3);
      }

      const conduit::Node &n_transforms = registered_filter_types()["transforms"];

      if(n_transforms.has_child(type))
      {
          filter_name = n_transforms[type].as_string();
      }
      else
      {
        ASCENT_ERROR("Unrecognized transform filter "<<filter["type"].as_string());
      }

      // create a unique name for the filter
      std::stringstream ss;
      ss<<pipeline_name<<"_"<<cname<<"_"<<type;
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

  const conduit::Node &n_extracts = registered_filter_types()["extracts"];

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
    filter_name = "ascent_python_script";

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
       std::string script_fname = params["file"].as_string();

       Node n_py_src;
       // read script only on rank 0
       if(m_rank == 0)
       {
         ostringstream py_src;
         ifstream ifs(script_fname.c_str());

         // guard by successful file open,
         // otherwise our prefix comment will always cause
         // a valid string to be passed to the node
         // and we won't be able to detect when there was
         // a bad file passed
         if(ifs.is_open())
         {
             py_src << "# script from: " << script_fname << std::endl;
             copy(istreambuf_iterator<char>(ifs),
                  istreambuf_iterator<char>(),
                  ostreambuf_iterator<char>(py_src));
             n_py_src.set(py_src.str());
             ifs.close();
         }
       }

       relay::mpi::broadcast_using_schema(n_py_src,0,comm);

       if(!n_py_src.dtype().is_string())
       {
         ASCENT_ERROR("failed to read python script file "
                      << script_fname
                      << " and broadcast source");
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
  else if(extract_type == "jupyter")
  {
    filter_name = "ascent_python_script";

    // customize the names of the script integration module and funcs
    params["interface/module"] = "ascent_extract";
    params["interface/input"]  = "ascent_data";
    params["interface/set_output"] = "ascent_set_output";

     ostringstream py_src_final;
#ifdef ASCENT_MPI_ENABLED
    // for MPI case, inspect args, if script is passed via file,
    // read contents on root and broadcast to other tasks
    int comm_id = flow::Workspace::default_mpi_comm();
    MPI_Comm comm = MPI_Comm_f2c(comm_id);
    // inject helper that provides the mpi comm handle

    py_src_final << "# ascent mpi comm helper function" << std::endl
                 << "def ascent_mpi_comm_id():" << std::endl
                 << "    return " << comm_id << std::endl
                 << std::endl
                 // bind ascent_mpi_comm_id into the module
                 << "ascent_extract.ascent_mpi_comm_id = ascent_mpi_comm_id"
                 << std::endl
                 // import our code that connects to the bridge kernel server
                 << "from ascent.mpi import jupyter_bridge "
                 << std::endl;
#else
    // import our code that connects to the bridge kernel server
    py_src_final << "from ascent import jupyter_bridge" << std::endl;
#endif
    // finally actually call the bridge kernel server
    py_src_final << "jupyter_bridge()" << std::endl;
    params["source"] = py_src_final.str();
  }
  // generic extract support
  else if(n_extracts.has_child(extract_type))
  {
     filter_name = n_extracts[extract_type].as_string();
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
    extract_source = "source";

  }
  m_connections[extract_name] = extract_source;

}
//-----------------------------------------------------------------------------
void
AscentRuntime::ConvertTriggerToFlow(const conduit::Node &trigger,
                                    const std::string trigger_name)
{
  std::string filter_name;

  conduit::Node params;
  if(trigger.has_path("params"))
  {
    params = trigger["params"];
  }

  std::string pipeline = "source";
  if(trigger.has_path("pipeline"))
  {
    pipeline = trigger["pipeline"].as_string();
  }

  w.graph().add_filter("basic_trigger",
                       trigger_name,
                       params);

  // this is the blueprint mesh
  m_connections[trigger_name] = pipeline;

}
//-----------------------------------------------------------------------------
void
AscentRuntime::ConvertQueryToFlow(const conduit::Node &query,
                                  const std::string query_name,
                                  const std::string prev_name)
{

  std::string filter_name;

  conduit::Node params;
  std::string pipeline = CreateDefaultFilters()["queries"].as_string();
  const std::string default_pipeline = pipeline;
  if(query.has_path("params"))
  {
    params = query["params"];
  }

  if(query.has_path("pipeline"))
  {
    pipeline = query["pipeline"].as_string();
  }


  w.graph().add_filter("basic_query",
                       query_name,
                       params);


  // connection port to enforce order of execution
  std::string conn_port;
  if(prev_name == "")
  {
    conn_port = pipeline;
  }
  else
  {
    conn_port = prev_name;
  }

  w.graph().connect(conn_port,
                    query_name,
                    "dummy");

  // this is the blueprint mesh
  m_connections[query_name] = pipeline;
  // we need all filters to depend on queries
  // from the source this keeps track of that
  if(pipeline == default_pipeline)
  {
    m_connections["ascent_last_query"] = query_name;
  }
}
//-----------------------------------------------------------------------------
void
AscentRuntime::ConvertPlotToFlow(const conduit::Node &plot,
                                 const std::string plot_name)
{
  std::string filter_name = "create_plot";

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
    plot_source = plot["pipeline"].as_string();
  }
  else
  {
    // default pipeline: directly connect to published data
    plot_source = CreateDefaultFilters()["filters"].as_string();
  }

  std::string pipeline_filter_name = plot_source;

  std::string prev_filter = plot_source;

#if defined(ASCENT_VTKM_ENABLED)
  const int num_ghosts = m_ghost_fields.number_of_children();
  if(num_ghosts != 0)
  {
    // we need to make sure that ghost zones don't make it into rendering
    // so we will create new filters that attach to the pipeline outputs
    std::string strip_name = plot_source + "_strip_real_ghosts";

    // we can have multiple ghost fields
    std::vector<std::string> ghost_fields;
    const int num_children = m_ghost_fields.number_of_children();
    for(int i = 0; i < num_children; ++i)
    {
      ghost_fields.push_back(m_ghost_fields.child(i).as_string());
    }

    const int num_ghosts = ghost_fields.size();
    for(int i = 0; i < num_ghosts; ++i)
    {
      std::string filter_name = strip_name + "_" + ghost_fields[i];

      // if this alread exists then it was created by another plot
      if(!w.graph().has_filter(filter_name))
      {
        conduit::Node threshold_params;
        threshold_params["field"] = ghost_fields[i];
        threshold_params["min_value"] = 0;
        threshold_params["max_value"] = 0;

        w.graph().add_filter("vtkh_ghost_stripper",
                             filter_name,
                             threshold_params);

        w.graph().connect(prev_filter,
                          filter_name,
                          0);        // default port

        prev_filter = filter_name;
      }
    }
  } // if stripping ghosts
#endif
  // create an a consistent name
  std::string endpoint_name = pipeline_filter_name + "_plot_source";

  if(!w.graph().has_filter(endpoint_name))
  {
    w.graph().add_filter("alias",
                         endpoint_name);

    w.graph().connect(prev_filter,   // src
                      endpoint_name, // dest
                      0);            // default port

  }

  m_connections[plot_name] = endpoint_name;

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
AscentRuntime::CreateTriggers(const conduit::Node &triggers)
{
  std::vector<std::string> names = triggers.child_names();
  for(int i = 0; i < triggers.number_of_children(); ++i)
  {
    conduit::Node trigger = triggers.child(i);
    ConvertTriggerToFlow(trigger, names[i]);
  }
}

//-----------------------------------------------------------------------------
void
AscentRuntime::CreateQueries(const conduit::Node &queries)
{
  std::vector<std::string> names = queries.child_names();
  std::string prev_name = "";
  for(int i = 0; i < queries.number_of_children(); ++i)
  {
    conduit::Node query = queries.child(i);
    ConvertQueryToFlow(query, names[i], prev_name);
    prev_name = names[i];
  }
}

//-----------------------------------------------------------------------------
void
AscentRuntime::PopulateMetadata()
{
  // add global state meta data to the registry
  const int num_domains = m_source.number_of_children();
  int cycle = -1;
  float time = -1.f;

  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = m_source.child(i);
    if(dom.has_path("state/cycle"))
    {
      cycle = dom["state/cycle"].to_int32();
    }
    if(dom.has_path("state/time"))
    {
      time = dom["state/time"].to_float32();
    }
  }


  if(cycle != -1)
  {
    Metadata::n_metadata["cycle"] = cycle;
  }
  if(time != -1.f)
  {
    Metadata::n_metadata["time"] = time;
  }

  Metadata::n_metadata["refinement_level"] = m_refinement_level;
  Metadata::n_metadata["ghost_field"] = m_ghost_fields;
  Metadata::n_metadata["default_dir"] = m_default_output_dir;

}
//-----------------------------------------------------------------------------
void
AscentRuntime::ConnectSource()
{
    // There is no promise that all data can be zero copied
    // and conversions to vtkh/low order will be invalid.
    // We must reset the source object
    conduit::Node *data_node = new conduit::Node();
    data_node->set_external(m_source);
    m_data_object.reset(data_node);

    SourceFieldFilter();

    // note: if the reg entry for data was already added
    // the set_external updates everything,
    // we don't need to remove and re-add.

    if(!w.registry().has_entry("_ascent_input_data"))
    {
        w.registry().add<DataObject>("_ascent_input_data",
                                     &m_data_object);
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

  // the default dummy connection is used to enforce the ordering
  // that ensures that source queries are available to filters
  // if not present, then there are no quieries and we just connect
  // the dummy port to another alias
  std::string default_dummy_connection = "default_queries_endpoint";

  std::vector<std::string> names = m_connections.child_names();
  for (int i = 0; i < m_connections.number_of_children(); ++i)
  {
    std::string pipeline = m_connections[names[i]].as_string();
    if(names[i] == "ascent_last_query")
    {
      // this is the name of the last query
      default_dummy_connection = pipeline;
      continue;
    }

    if(pipeline == "default")
    {
      pipeline = CreateDefaultFilters()["filters"].as_string();
    }
    else if(!w.graph().has_filter(pipeline))
    {
      ASCENT_ERROR(names[i]<<"' references unknown pipeline: "<<pipeline);
    }

    w.graph().connect(pipeline, // src
                      names[i], // dest
                      0);       // default port
  }

  if(w.graph().has_filter("default_filters_endpoint"))
  {
    // now connect the dummy port of the default_filters
    w.graph().connect(default_dummy_connection,   // src
                      "default_filters_endpoint", // dest
                      1);                         // dummy port
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
      pipeline = CreateDefaultFilters()["filters"].as_string();
    }
    // use the consistent name from PlotToFlow
    pipeline = pipeline + "_plot_source";
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

    if(scene.has_path("image_prefix") || scene.has_path("image_name"))
    {
      if(scene.has_path("image_prefix"))
      {
        render_params["image_prefix"] = scene["image_prefix"].as_string();
      }
      else
      {
        render_params["image_name"] = scene["image_name"].as_string();
      }
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

    // create plots with scene name + plot_name
    conduit::Node appended_plots;
    for(int k = 0; k < plot_names.size(); ++k)
    {
      std::string p_name = names[i] + "_" + plot_names[k];
      appended_plots[p_name] = scene["plots/" + plot_names[k]];
    }

    plot_names = appended_plots.child_names();

    CreatePlots(appended_plots);

    std::vector<std::string> bounds_names;
    std::vector<std::string> union_bounds_names;

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
      // A render, needs one inputs
      //     1) the coordinate bounds of all the input pipelines
      //        to be able to create a defailt camera
      // Thus, we have to call bounds and
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

        if(p == 1)
        {
          // first union just needs the output
          // of the first bounds
          w.graph().connect(bounds_names[p-1],  // src
                            union_bounds_name,  // dest
                            0);                 // default port
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

        }

        w.graph().connect(bounds_name,        // src
                          union_bounds_name,  // dest
                          1);                 // default port

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

    if(bounds_names.size() == 1)
    {
      bounds_output = bounds_names[0];
    }
    else
    {
      const size_t union_size = union_bounds_names.size();
      bounds_output = union_bounds_names[union_size-1];
    }

    w.graph().connect(bounds_output, // src
                      renders_name,  // dest
                      0);            // default port

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

//-----------------------------------------------------------------------------
void
AscentRuntime::FindRenders(conduit::Node &image_params,
                           conduit::Node& image_list)
{
    image_list.reset();

    if(!w.registry().has_entry("image_list"))
    {
      return;
    }

    Node *images = w.registry().fetch<Node>("image_list");

    const int size = images->number_of_children();
    image_params = *images;
    for(int i = 0; i < size; i++)
    {
      image_list.append() = images->child(i)["image_name"].as_string();
    }

    images->reset();

}

//-----------------------------------------------------------------------------
void
AscentRuntime::BuildGraph(const conduit::Node &actions)
{
  // make sure to clean helpers that we use
  // to build the graph
  m_connections.reset();
  m_scene_connections.reset();
  m_save_session_actions.reset();

  // execution will be enforced in the following order:
  conduit::Node queries;
  conduit::Node triggers;
  conduit::Node pipelines;
  conduit::Node scenes;
  conduit::Node extracts;

  // Loop over the actions
  for (int i = 0; i < actions.number_of_children(); ++i)
  {
      const Node &action = actions.child(i);
      if(!action.has_path("action"))
      {
        ASCENT_ERROR("Malformed actions");
      }
      string action_name = action["action"].as_string();

      if(action_name == "add_pipelines")
      {
        if(action.has_path("pipelines"))
        {
          pipelines.append() = action["pipelines"];
        }
        else
        {
          ASCENT_ERROR("action 'add_pipelines' missing child 'pipelines'");
        }
      }
      else if(action_name == "add_scenes")
      {
        if(action.has_path("scenes"))
        {
          scenes.append() = action["scenes"];
        }
        else
        {
          ASCENT_ERROR("action 'add_scenes' missing child 'scenes'");
        }
      }
      else if(action_name == "add_extracts")
      {
        if(action.has_path("extracts"))
        {
          extracts.append() = action["extracts"];
        }
        else
        {
          ASCENT_ERROR("action 'add_extracts' missing child 'extracts'");
        }
      }
      else if(action_name == "add_triggers")
      {
        if(action.has_path("triggers"))
        {
          triggers.append() = action["triggers"];
        }
        else
        {
          ASCENT_ERROR("action 'add_triggers' missing child 'triggers'");
        }
      }
      else if(action_name == "add_queries")
      {
        if(action.has_path("queries"))
        {
          queries.append() = action["queries"];
        }
        else
        {
          ASCENT_ERROR("action 'add_queries' missing child 'queries'");
        }
      }
      else if( action_name == "execute" ||
               action_name == "reset")
      {
        // These actions are now deprecated. To avoid
        // issues with existing integrations we will just
        // do nothing
      }
      else if(action_name == "save_session")
      {
        // Saving the session will be deferred to after
        // the workspace executes.
        m_save_session_actions.append() = action;
      }
      else
      {
        ASCENT_ERROR("Unknown action ' "<<action_name<<"'");
      }

  }

  for(int i = 0; i < pipelines.number_of_children(); ++i)
  {
    CreatePipelines(pipelines.child(i));
  }
  for(int i = 0; i < queries.number_of_children(); ++i)
  {
    CreateQueries(queries.child(i));
  }
  for(int i = 0; i < triggers.number_of_children(); ++i)
  {
    CreateTriggers(triggers.child(i));
  }
  for(int i = 0; i < scenes.number_of_children(); ++i)
  {
    CreateScenes(scenes.child(i));
  }
  for(int i = 0; i < extracts.number_of_children(); ++i)
  {
    CreateExtracts(extracts.child(i));
  }

  ConnectGraphs();
}
//-----------------------------------------------------------------------------
void
AscentRuntime::Execute(const conduit::Node &actions)
{
    bool log_timings = false;
    if(m_runtime_options.has_child("timings") &&
       m_runtime_options["timings"].as_string() == "true")
    {
      log_timings = true;
    }

    w.enable_timings(log_timings);

    // catch any errors that come up here and forward
    // them up as a conduit error

    // --- open try --- //
    try
    {
        ResetInfo();

        conduit::Node diff_info;
        bool different_actions = m_previous_actions.diff(actions, diff_info);

        if(different_actions)
        {
          if(m_field_filtering)
          {
            // check to see if we can determine what
            // fields the actions need
            conduit::Node info;
            bool success = field_list(actions, m_field_list, info);
            if(!success)
            {
              ASCENT_ERROR("Field filtering failed: "<<info.to_yaml());
            }
            if(m_field_list.size() == 0)
            {
              ASCENT_ERROR("Field filtering failed to find any fields");
            }
          }

          // destroy existing graph an start anew
          w.reset();
          ConnectSource();
          BuildGraph(actions);
        }
        else
        {
          // always ensure that we have the source
          ConnectSource();
        }


        m_previous_actions = actions;

        PopulateMetadata(); // add metadata so filters can access it

        // add the source to the registry so we can access information
        // about the original mesh (like bounds)
        w.registry().add<DataObject>("source_object", &m_data_object,1);

        w.info(m_info["flow_graph"]);
        m_info["actions"] = actions;
        // w.print();
        // std::cout<<w.graph().to_dot();
        //w.graph().save_dot_html("ascent_flow_graph.html");

#if defined(ASCENT_VTKM_ENABLED)
        if(log_timings)
        {
          int cycle = 0;
          if(Metadata::n_metadata.has_path("cycle"))
          {
            cycle = Metadata::n_metadata["cycle"].to_int32();
          }
          std::stringstream ss;
          ss<<"cycle_"<<cycle;
          vtkh::DataLogger::GetInstance()->OpenLogEntry(ss.str());
          vtkh::DataLogger::GetInstance()->AddLogData("cycle", cycle);
        }
#endif
        // now execute the data flow graph
        w.execute();

#if defined(ASCENT_VTKM_ENABLED)
        if(log_timings)
        {
          vtkh::DataLogger::GetInstance()->CloseLogEntry();
        }
#endif
        if(m_save_session_actions.number_of_children() > 0)
        {
          SaveSession();
        }

        Node msg;
        this->Info(msg["info"]);
        ascent::about(msg["about"]);
        m_web_interface.PushMessage(msg);

        // add render results to info
        Node render_file_names;
        Node renders;
        FindRenders(renders, render_file_names);

        if(renders.number_of_children() > 0)
        {
            m_info["images"] = renders;
        }

        // add extract results to info
        if(w.registry().has_entry("extract_list"))
        {
            Node *extracts_list = w.registry().fetch<Node>("extract_list");
            if(extracts_list->number_of_children() > 0)
            {
                m_info["extracts"].set(*extracts_list);
            }
            // always clear after fetch.
            extracts_list->reset();
        }

        // add expression results to info
        const conduit::Node &expression_cache =
          runtime::expressions::ExpressionEval::get_cache();

        if(expression_cache.number_of_children() > 0)
        {
          runtime::expressions::ExpressionEval::get_last(m_info["expressions"]);
        }

        // add flow graphviz details to info
        m_info["flow_graph_dot"]      = w.graph().to_dot();
        m_info["flow_graph_dot_html"] = w.graph().to_dot_html();

        m_web_interface.PushRenders(render_file_names);

        w.registry().reset();
    }
    // --- close try --- //

    // Defend calling code by repackaging
    // as many errors as possible with catch statements
#if defined(ASCENT_VTKM_ENABLED)
    // bottle vtkm and vtkh errors
    catch(vtkh::Error &e)
    {
      w.reset();
      ASCENT_ERROR("Execution failed with vtkh: "<<e.what());
    }
    catch(vtkm::cont::Error &e)
    {
      w.reset();
      ASCENT_ERROR("Execution failed with vtkm: "<<e.what());
    }
#endif
    catch(conduit::Error &e)
    {
      w.reset();
      throw e;
    }
    catch(std::exception &e)
    {
      w.reset();
      std::cerr<<"Execution failed with exception: "<<e.what()<<"\n";
    }
    catch(...)
    {
      w.reset();
      ASCENT_ERROR("Ascent: unknown exception thrown");
    }
}

//-----------------------------------------------------------------------------
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
void AscentRuntime::SourceFieldFilter()
{
  if(!m_field_filtering)
  {
    return;
  }

  bool high_order = m_data_object.source() == DataObject::Source::HIGH_BP;
  conduit::Node *data = m_data_object.as_node().get();
  const int num_domains = data->number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = data->child(i);
    if(dom.has_path("fields"))
    {

      const int num_fields = dom["fields"].number_of_children();
      std::vector<std::string> names = dom["fields"].child_names();
      for(int f = 0; f < num_fields; ++f)
      {
        if(high_order)
        {
          // handle special mfem fields
          if(names[f].find("position") != std::string::npos ||
             names[f].find("_nodes") != std::string::npos ||
             names[f].find("_attribute") != std::string::npos ||
             names[f].find("boundary") != std::string::npos)
          {
            continue;
          }
        }
        if(std::find(m_field_list.begin(),
                     m_field_list.end(),
                     names[f]) == m_field_list.end())
        {
            // remove the field
            dom.remove("fields/"+names[f]);
        }
      } // for fields
    }
  } // for doms

}


//-----------------------------------------------------------------------------
void AscentRuntime::PaintNestsets()
{
  std::vector<std::string> ghosts;
  std::map<std::string,std::string> topo_ghosts;
  std::map<std::string,std::string> topo_nestsets;

  bool bad_bp = false;

  const int num_ghosts = m_ghost_fields.number_of_children();

  for(int i = 0; i < num_ghosts; ++i)
  {
    ghosts.push_back(m_ghost_fields.child(i).as_string());
  }
  // we have to be careful since we have not called verify
  // but we need to do this now, since we might be creating
  // new ghost fields

  const int num_domains = m_source.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = m_source.child(i);

    if(dom.has_path("nestsets"))
    {
      const conduit::Node &nestsets = dom["nestsets"];
      const int num_nests = nestsets.number_of_children();
      std::vector<std::string> nest_names = nestsets.child_names();
      for(int n = 0; n < num_nests; ++n)
      {
        const conduit::Node &nestset = nestsets.child(n);
        if(nestset.has_path("topology"))
        {
          topo_nestsets[nestset["topology"].as_string()] = nest_names[n];
        }
        else
        {
          // maybe we just do nothing and let verify tell people
          // about the sins of the mesh
          bad_bp = true;
          break;
        }

      }
    }

    for(int f = 0; f < num_ghosts; ++f)
    {
      const string fpath = "fields/"+ghosts[f];
      if(dom.has_path(fpath))
      {
        if(dom.has_path(fpath+"/topology"))
        {
          topo_ghosts[dom[fpath+"/topology"].as_string()] = ghosts[f];
        }
        else
        {
          bad_bp = true;
          break;
        }
      }
    }
  }
  if(bad_bp)
  {
    // its bad, punt reporting to verify;
    return;
  }

  // ok, we have built up a map of topologies, nestsets, and ghosts;
  // if we have a ghost that is associated with a topology with a nestset,
  // then we will augment the ghost field with 1s(real ghosts) in zones
  // that are covered by a finer resolution mesh, that are not already
  // marked as ghosts.
  // If there arent't ghosts associated with a nestset topology,
  // we will create them.
  std::set<std::string> new_ghosts;

  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = m_source.child(i);
    const int num_topos = dom["topologies"].number_of_children();
    const std::vector<std::string> topo_names = dom["topologies"].child_names();
    for(auto topo_name : topo_names)
    {
      bool has_ghost = topo_ghosts.find(topo_name) != topo_ghosts.end();
      bool has_nest = topo_nestsets.find(topo_name) != topo_nestsets.end();

      if(!has_nest)
      {
        continue;
      }

      std::string nest_name = topo_nestsets[topo_name];

      if(has_ghost)
      {
        std::string ghost_name = topo_ghosts[topo_name];
        if(dom.has_path("fields/" + ghost_name))
        {
          // ok, we need to alter the ghosts but the simulation
          // gave us this data. In most cases, the ascent
          // integration made the ghost zones, so it would
          // be safe to change them. That said, it would
          // be bad practice to alter the data, so we will make a
          // copy and update our tree to point at the copy.
          const std::string ghost_path = "fields/" + ghost_name;
          const std::string temp_path = "fields/" + ghost_name + "_bananas";

          conduit::Node &field = dom["fields/" + ghost_name];
          dom[temp_path].set(field);
          dom.remove(ghost_path);
          //dom.rename_child(temp_path, ghost_path);
          dom["fields/"].rename_child(ghost_name+"_bananas", ghost_name);

          conduit::Node &ghost_field = dom[ghost_path];

          runtime::expressions::paint_nestsets(nest_name, topo_name,  dom, ghost_field);
        }
        else
        {
          // this is weird and shouldn't happen in the real world.
          // Some domain had a ghost but others didn't
          ASCENT_ERROR("missing ghost field "<<ghost_name);
        }
      }
      else
      {
        // there are no ghosts, so we have to build a new field
        std::string ghost_name = topo_name + "_ghosts";
        conduit::Node &field = dom["fields/" + ghost_name];
        field.reset();
        runtime::expressions::paint_nestsets(nest_name, topo_name, dom, field);
        new_ghosts.insert(ghost_name);
      }
    }
  }

  for(auto name : new_ghosts)
  {
    ASCENT_INFO("added new ghost field because of nestset: "<<name);
    m_ghost_fields.append() = name;
  }

}

void AscentRuntime::VerifyGhosts()
{
  conduit::Node verified;
  const int num_ghosts = m_ghost_fields.number_of_children();
  for(int i = 0; i < num_ghosts; ++i)
  {
    std::string ghost_name = m_ghost_fields.child(i).as_string();
    if(runtime::expressions::has_field(m_source, ghost_name))
    {
      verified.append() = ghost_name;
    }
    else
    {
      // only report errors for user defined ghosts
      if(ghost_name != "ascent_ghosts")
      {
        std::stringstream ss;
        if(m_source.number_of_children() > 0)
        {
          if(m_source.child(0).has_path("fields"))
          {
            std::vector<string> names = m_source.child(0)["fields"].child_names();
            for(auto name : names)
            {
              ss<<" '"<<name<<"'";
            }
          }
          else
          {
            ss<<"can't deduce possible fields. "
              <<"Published data does not contain fields in dom 0";
          }
        }
        ASCENT_ERROR("User specified Ghost field '"<<ghost_name
                     <<"' does not exist. Possible fields: "
                     <<ss.str());
      }
    }
  }
  m_ghost_fields = verified;

}

void AscentRuntime::SaveSession()
{
  const int num_actions = m_save_session_actions.number_of_children();

  for(int a = 0; a < num_actions; ++a)
  {
    const conduit::Node &action = m_save_session_actions.child(a);

    std::string filename = m_session_name;
    if(action.has_path("file_name"))
    {
      if(!action["file_name"].dtype().is_string())
      {
        ASCENT_ERROR("save_session filename must be a string");
      }
      filename = action["file_name"].as_string();
    }

    // allow the user to specify which expressions they want saved out
    if(action.has_path("expressions"))
    {
      std::vector<std::string> expressions_selection;
      const conduit::Node &elist = action["expressions"];
      const int num_exprs= elist.number_of_children();
      if(num_exprs == 0)
      {
        ASCENT_ERROR("save_session expression selection must be "
                     <<" a non-empty list of strings");
      }

      for(int i = 0; i < num_exprs; ++i)
      {
        const conduit::Node &e = elist.child(i);
        if(!e.dtype().is_string())
        {
           ASCENT_ERROR("save_session expression selection list "
                        <<"values must be a string");
        }
        expressions_selection.push_back(e.as_string());
      }
      runtime::expressions::ExpressionEval::save_cache(filename, expressions_selection);
    }
    else
    {
      runtime::expressions::ExpressionEval::save_cache(filename);
    }
  } // for each save action
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



