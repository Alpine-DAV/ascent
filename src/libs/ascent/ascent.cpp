//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent.cpp
///
//-----------------------------------------------------------------------------

#include <ascent.hpp>
#include <ascent_license.hpp>
#include <ascent_runtime.hpp>

#include <ascent_empty_runtime.hpp>
#include <ascent_flow_runtime.hpp>
#include <runtimes/ascent_main_runtime.hpp>
#include <utils/ascent_string_utils.hpp>
#include <flow.hpp>

#if defined(ASCENT_VTKH_ENABLED)
    #include <vtkh/vtkh.hpp>
#endif

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#endif
using namespace conduit;
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
void
quiet_handler(const std::string &,
              const std::string &,
              int )
{
}

//-----------------------------------------------------------------------------
Ascent::Ascent()
: m_runtime(NULL),
  m_verbose_msgs(true),
  m_forward_exceptions(false),
  m_actions_file("<<UNSET>>")
{
  m_options["mpi_comm"] = -1;
  set_status("Ascent instance created");
}

//-----------------------------------------------------------------------------
Ascent::~Ascent()
{

}

//-----------------------------------------------------------------------------
void
Ascent::open()
{
    Node opts;
    open(opts);
}

//-----------------------------------------------------------------------------
void
CheckForSettingsFile(std::string file_name,
                     conduit::Node &node,
                     bool merge,
                     int mpi_comm_id)
{
    int comm_size = 1;
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    if(mpi_comm_id == -1)
    {
      // do nothing, an error will be thrown later
      // so we can respect the exception handling
      return;
    }
    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_id);
    MPI_Comm_size(mpi_comm, &comm_size);
    MPI_Comm_rank(mpi_comm, &rank);
#endif
    int has_file = 0;
    if(rank == 0 && conduit::utils::is_file(file_name))
    {
      has_file = 1;
    }
#ifdef ASCENT_MPI_ENABLED
    MPI_Bcast(&has_file, 1, MPI_INT, 0, mpi_comm);
#endif
    if(has_file == 0)
    {
      return;
    }

    int actions_file_valid = 0;
    std::string emsg = "";

    if(rank == 0)
    {
      std::string curr,next;

      std::string protocol = "json";
      // if file ends with yaml, use yaml as proto
      conduit::utils::rsplit_string(file_name,
                                    ".",
                                    curr,
                                    next);

      if(curr == "yaml")
      {
        protocol = "yaml";
      }

      try
      {
        conduit::Node file_node;
        file_node.load(file_name, protocol);

        if(merge)
        {
          node.update(file_node);
        }
        else
        {
          node = file_node;
        }

        actions_file_valid = 1;
      }
      catch(conduit::Error &e)
      {
        // failed to open or parse the actions file
        actions_file_valid = 0;
        emsg = e.message();
      }
    }

#ifdef ASCENT_MPI_ENABLED
    // make sure all ranks error if the parsing on rank 0 failed.
    MPI_Bcast(&actions_file_valid, 1, MPI_INT, 0, mpi_comm);
#endif

    if(actions_file_valid == 0)
    {
        // Raise Error
        ASCENT_ERROR("Failed to load actions file: " << file_name
                     << "\n" << emsg);
    }
#ifdef ASCENT_MPI_ENABLED
    relay::mpi::broadcast_using_schema(node, 0, mpi_comm);
#endif
}

//-----------------------------------------------------------------------------
void
Ascent::open(const conduit::Node &options)
{
    try
    {
        if(m_runtime != NULL)
        {
            ASCENT_ERROR("Ascent Runtime already initialized!");
        }

        std::string opts_file = "ascent_options.json";

        if(!conduit::utils::is_file(opts_file))
        {
            opts_file = "ascent_options.yaml";
        }

        Node processed_opts(options);

        int comm_id = -1;
        if(options.has_path("mpi_comm"))
        {
          comm_id = options["mpi_comm"].to_int32();
        }

        CheckForSettingsFile(opts_file,
                             processed_opts,
                             true,
                             comm_id);

        m_options = processed_opts;

        // gaurd against funky things happening in the
        // user provided options
        if(options.has_path("mpi_comm"))
        {
          m_options["mpi_comm"] = options["mpi_comm"];
        }

        if(m_options.has_path("messages") &&
           m_options["messages"].dtype().is_string() )
        {
            std::string msgs_opt = m_options["messages"].as_string();
            if( msgs_opt == "verbose")
            {
                m_verbose_msgs = true;
            }
            else if(msgs_opt == "quiet")
            {
                m_verbose_msgs = false;
            }
        }

        if(m_options.has_path("exceptions") &&
           m_options["exceptions"].dtype().is_string() )
        {
            std::string excp_opt = m_options["exceptions"].as_string();
            if( excp_opt == "catch")
            {
                m_forward_exceptions = false;
            }
            else if(excp_opt == "forward")
            {
                m_forward_exceptions = true;
            }
        }

        if(m_options.has_path("actions_file") &&
           m_options["actions_file"].dtype().is_string() )
        {
            m_actions_file = m_options["actions_file"].as_string();
        }


        Node cfg;
        ascent::about(cfg);

        std::string runtime_type = cfg["default_runtime"].as_string();

        if(m_options.has_path("runtime"))
        {
            if(m_options.has_path("runtime/type"))
            {
                runtime_type = m_options["runtime/type"].as_string();
            }
        }

        if(runtime_type == "empty")
        {
            m_runtime = new EmptyRuntime();
        }
        else if(runtime_type == "ascent")
        {
            m_runtime = new AscentRuntime();
            if(m_options.has_path("runtime/vtkm/backend"))
            {
    #if defined(ASCENT_VTKH_ENABLED)
              std::string backend = m_options["runtime/vtkm/backend"].as_string();
              if(backend == "serial")
              {
                vtkh::ForceSerial();
              }
              else if(backend == "openmp")
              {
                vtkh::ForceOpenMP();
              }
              else if(backend == "cuda")
              {
                vtkh::ForceCUDA();
              }
              else
              {
                ASCENT_ERROR("Ascent unrecognized backend "<<backend);
              }
    #else
              ASCENT_ERROR("Ascent vtkm backend is disabled. "
                          "Ascent was not built with vtk-m support");
    #endif
            }
        }
        else if(runtime_type == "flow")
        {
            m_runtime = new FlowRuntime();
        }
        else
        {
            ASCENT_ERROR("Unsupported Runtime type "
                           << "\"" << runtime_type << "\""
                           << " passed via 'runtime' open option.");
        }

        m_runtime->Initialize(m_options);

        // don't print info messages unless we are using verbose
        // Runtimes may set their own handlers in initialize, so
        // make sure to do this after.
        if(!m_verbose_msgs)
        {
            conduit::utils::set_info_handler(quiet_handler);
        }

        set_status("Ascent::open completed");
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::open failed",
                    e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
            std::cout << "[Error] Ascent::open "
                      << e.message() << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
void
Ascent::register_callback(const std::string &callback_name,
                          void (*callback_function)(void))
{
     try
    {
        if(m_runtime != NULL)
        {
            m_runtime->RegisterCallback(callback_name, callback_function);
        }
        else
        {
            ASCENT_ERROR("Ascent Runtime is not initialized");
        }

        set_status("Ascent::register_callback completed");
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::register_callback failed",
                   e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
          if(m_runtime != NULL)
          {
            std::stringstream msg;
            msg << "[Error] Ascent::register_callback "
                << e.message() << std::endl;
            m_runtime->DisplayError(msg.str());
          }
          else
          {
            std::cerr<< "[Error] Ascent::register_callback "
                    << e.message() << std::endl;
          }
        }
    }
}

//-----------------------------------------------------------------------------
void
Ascent::register_callback(const std::string &callback_name,
                          bool (*callback_function)(void))
{
     try
    {
        if(m_runtime != NULL)
        {
            m_runtime->RegisterCallback(callback_name, callback_function);
        }
        else
        {
            ASCENT_ERROR("Ascent Runtime is not initialized");
        }

        set_status("Ascent::register_callback completed");
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::register_callback failed",
                   e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
          if(m_runtime != NULL)
          {
            std::stringstream msg;
            msg << "[Error] Ascent::register_callback "
                << e.message() << std::endl;
            m_runtime->DisplayError(msg.str());
          }
          else
          {
            std::cerr<< "[Error] Ascent::register_callback "
                    << e.message() << std::endl;
          }
        }
    }
}

//-----------------------------------------------------------------------------
void
Ascent::publish(const conduit::Node &data)
{
    try
    {
        if(m_runtime != NULL)
        {
            m_runtime->Publish(data);
        }
        else
        {
            ASCENT_ERROR("Ascent Runtime is not initialized");
        }

        set_status("Ascent::publish completed");
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::publish failed",
                   e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
          if(m_runtime != NULL)
          {
            std::stringstream msg;
            msg << "[Error] Ascent::publish "
                << e.message() << std::endl;
            m_runtime->DisplayError(msg.str());
          }
          else
          {
            std::cerr<< "[Error] Ascent::publish "
                    << e.message() << std::endl;
          }
        }
    }
}

//-----------------------------------------------------------------------------
void
Ascent::execute(const conduit::Node &actions)
{
    try
    {
        if(m_runtime != NULL)
        {
            Node processed_actions(actions);

            if(m_actions_file == "<<UNSET>>")
            {
                m_actions_file = "ascent_actions.json";

                if(!conduit::utils::is_file(m_actions_file))
                {
                    m_actions_file = "ascent_actions.yaml";
                }
            }
            else if(m_actions_file != "ascent_actions.json" &&
                    m_actions_file != "ascent_actions.yaml" &&
                    m_actions_file != "")
            {
                // an actions file has been set by the user
                // so we better let them know if we don't find
                // it
                if(!conduit::utils::is_file(m_actions_file))
                {
                    ASCENT_ERROR("An actions file '"
                                 <<m_actions_file<<"' was specified "
                                 " but could not be found. Please "
                                 "check if the file is in the current "
                                 "directory or provide an absolute path.")
                }
            }

            CheckForSettingsFile(m_actions_file,
                                 processed_actions,
                                 false,
                                 m_options["mpi_comm"].to_int32());

            m_runtime->Execute(processed_actions);

            set_status("Ascent::execute completed");
        }
        else
        {
            ASCENT_ERROR("Ascent Runtime is not initialized");
        }
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::execute failed",
                   e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
          if(m_runtime != NULL)
          {
            std::stringstream msg;
            msg << "[Error] Ascent::execute "
                << e.message() << std::endl;
            m_runtime->DisplayError(msg.str());
          }
          else
          {
            std::cerr<< "[Error] Ascent::execute "
                     << e.message() << std::endl;
          }
        }
    }
}


//-----------------------------------------------------------------------------
void
Ascent::info(conduit::Node &info_out)
{
    try
    {
        if(m_runtime != NULL)
        {
            m_runtime->Info(info_out);
        }

        info_out["status"] = m_status;

        // this doesn't modify status unless
        // info triggers an error
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::info failed",
                   e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
          if(m_runtime != NULL)
          {
            std::stringstream msg;
            msg << "[Error] Ascent::info"
                << e.message() << std::endl;
            m_runtime->DisplayError(msg.str());
          }
          else
          {
            std::cerr<< "[Error] Ascent::info"
                     << e.message() << std::endl;
          }
        }
    }
}


//-----------------------------------------------------------------------------
conduit::Node &
Ascent::info()
{
    try
    {
        if(m_runtime == NULL)
        {
            
        }
        else // we don't have info throw and error
        {
            conduit::Node &info = m_runtime->Info();
            info["status"].set(m_status);
            return info;
        }
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::info failed",
                   e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
          if(m_runtime != NULL)
          {
            std::stringstream msg;
            msg << "[Error] Ascent::info"
                << e.message() << std::endl;
            m_runtime->DisplayError(msg.str());
          }
          else
          {
            std::cerr<< "[Error] Ascent::info"
                     << e.message() << std::endl;
          }
        }
    }

    m_info.reset();
    m_info["status"] = m_status;
    return m_info;
}

//-----------------------------------------------------------------------------
void
Ascent::close()
{
    try
    {
        if(m_runtime != NULL)
        {
            delete m_runtime;
            m_runtime = NULL;
        }

         set_status("Ascent::close completed");
    }
    catch(conduit::Error &e)
    {
        set_status("Ascent::close failed",
                   e.message());

        if(m_forward_exceptions)
        {
            throw e;
        }
        else
        {
            if(m_runtime != NULL)
            {
              std::stringstream msg;
              msg << "[Error] Ascent::close"
                  << e.message() << std::endl;
              m_runtime->DisplayError(msg.str());
            }
            else
            {
              std::cerr<< "[Error] Ascent::close "
                        << e.message() << std::endl;
            }
        }
    }
}

//---------------------------------------------------------------------------//
void
Ascent::set_status(const std::string &msg)
{
    m_status.reset();
    std::ostringstream oss;
    oss << msg << " at " << timestamp();
    m_status["message"] = oss.str();
}

//---------------------------------------------------------------------------//
void
Ascent::set_status(const std::string &msg,
                   const std::string &details)
{
    m_status.reset();
    std::ostringstream oss;
    oss << msg << " at " << timestamp();
    m_status["message"] = oss.str();
    m_status["details"] = details;
}

//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    ascent::about(n);

    std::string ASCENT_MASCOT = "\n"
    "                                       \n"
    "         &&&&&&&&&&&                   \n"
    "       &&%&&@@@@@&&&%&&                \n"
    "      &%%&@       #@&%%&@              \n"
    "      %%%@           @&%%%%&           \n"
    "      %%%,            &%%%%%%%&        \n"
    "      &%#%*          &%##%######%%%%%  \n"
    "       @%###%&       %####%&%%%#####   \n"
    "          %###############&    @%##    \n"
    "                %%@*%((((##      &     \n"
    "                   ,#(((((#            \n"
    "                   @(////(%            \n"
    "                   &(/////#            \n"
    "                   &(/***/#            \n"
    "       #&&%%%#####%&(****/(            \n"
    "   %(////***********(*****(            \n"
    "  (********,,,,,,,**(*,,,*(            \n"
    " #**,,,*************%*,,,*(            \n"
    " (*,,,*/(((((((((#% %*,,,*%            \n"
    " /,.,*#              /,.,*             \n"
    " (,..,(             ,(,..,             \n"
    " %,..,(              (,..,             \n"
    "  ,..,/              #,..,             \n"
    "  *..,(              %...,             \n"
    "  /..,(              ..  ,             \n"
    "  @. ,#               .  .             \n"
    "  (..,#               .  .             \n"
    "\n\n"
    "Ascent Mascot ASCII Art is licensed under the: \n"
    " Creative Commons - Attribution - Share Alike license.\n"
    "  https://creativecommons.org/licenses/by-sa/3.0/\n"
    "\n"
    " Derived from:\n"
    "  https://www.thingiverse.com/thing:5340\n";

    return n.to_yaml() + "\n" + ASCENT_MASCOT;

}

//---------------------------------------------------------------------------//
void
about(conduit::Node &n)
{
    n.reset();
    n["version"] = ASCENT_VERSION;

#ifdef ASCENT_GIT_SHA1
    n["git_sha1"] = ASCENT_GIT_SHA1;
#else
    n["git_sha1"] = "unknown";
#endif

#ifdef ASCENT_GIT_SHA1_ABBREV
    n["git_sha1_abbrev"] = ASCENT_GIT_SHA1_ABBREV;
#else
    n["git_sha1_abbrev"] = "unknown";
#endif

#ifdef ASCENT_GIT_TAG
    n["git_tag"] = ASCENT_GIT_TAG;
#else
    n["git_tag"] = "unknown";
#endif

    if(n["git_tag"].as_string() == "unknown" &&
       n["git_sha1_abbrev"].as_string() != "unknown")
    {
        n["version"] = n["version"].as_string()
                       + "-" + n["git_sha1_abbrev"].as_string();
    }

    n["compilers/cpp"] = ASCENT_CPP_COMPILER;
#ifdef ASCENT_FORTRAN_COMPILER
    n["compilers/fortran"] = ASCENT_FORTRAN_COMPILER;
#endif

#if   defined(ASCENT_PLATFORM_WINDOWS)
    n["platform"] = "windows";
#elif defined(ASCENT_PLATFORM_APPLE)
    n["platform"] = "apple";
#else
    n["platform"] = "linux";
#endif

    n["system"] = ASCENT_SYSTEM_TYPE;
    n["install_prefix"] = ASCENT_INSTALL_PREFIX;
    n["license"] = ASCENT_LICENSE_TEXT;

// caliper annotations support
#if defined(ASCENT_CALIPER_ENABLED)
    n["annotations"] = "enabled";
#else 
    n["annotations"] = "disabled";
#endif

    std::string install_prefix = n["install_prefix"].as_string();
    std::string web_root = utils::join_file_path(install_prefix,"share");
    web_root = utils::join_file_path(web_root,"ascent");
    web_root = utils::join_file_path(web_root,"web_clients");
    n["web_client_root"] =  web_root;

#if defined(ASCENT_MPI_ENABLED)
    n["mpi"] = "enabled";
#else
    n["mpi"] = "disabled";
#endif

#if defined(ASCENT_OPENMP_ENABLED)
    n["openmp"] = "enabled";
#else
    n["openmp"] = "disabled";
#endif

#if defined(ASCENT_CUDA_ENABLED)
    n["cuda"] = "enabled";
#else
    n["cuda"] = "disabled";
#endif

#if defined(ASCENT_HIP_ENABLED)
    n["hip"] = "enabled";
#else
    n["hip"] = "disabled";
#endif

    // we will always have the main runtime available
    n["runtimes/ascent/status"] = "enabled";

// optional runtime eatures

// raja
#if defined(ASCENT_RAJA_ENABLED)
    n["runtimes/ascent/raja/status"] = "enabled";
#else
    n["runtimes/ascent/raja/status"] = "disabled";
#endif

// umpire
#if defined(ASCENT_UMPIRE_ENABLED)
    n["runtimes/ascent/umpire/status"] = "enabled";
#else
    n["runtimes/ascent/umpire/status"] = "disabled";
#endif

// dray
#if defined(ASCENT_DRAY_ENABLED)
    n["runtimes/ascent/dray/status"] = "enabled";
#else
    n["runtimes/ascent/dray/status"] = "disabled";
#endif

// occa jit
#if defined(ASCENT_JIT_ENABLED)
    n["runtimes/ascent/jit/status"] = "enabled";
#else
    n["runtimes/ascent/jit/status"] = "disabled";
#endif
    
    
// vtk-m + vtk-h
#if defined(ASCENT_VTKH_ENABLED)
    // call this vtkm so people don't have to know
    // about vtkh
    n["runtimes/ascent/vtkm/status"] = "enabled";
    if(vtkh::IsSerialEnabled())
    {
        n["runtimes/ascent/vtkm/backends/serial"] = "enabled";
    }
    else
    {
        n["runtimes/ascent/vtkm/backends/serial"] = "disabled";
    }

    if(vtkh::IsOpenMPEnabled())
    {
        n["runtimes/ascent/vtkm/backends/openmp"] = "enabled";
    }
    else
    {
        n["runtimes/ascent/vtkm/backends/openmp"] = "disabled";
    }

    if(vtkh::IsCUDAEnabled())
    {
        n["runtimes/ascent/vtkm/backends/cuda"] = "enabled";
    }
    else
    {
        n["runtimes/ascent/vtkm/backends/cuda"] = "disabled";
    }
    if(vtkh::IsKokkosAvailable())
    {
        n["runtimes/ascent/vtkm/backends/kokkos"] = "enabled";
    }
    else
    {
        n["runtimes/ascent/vtkm/backends/kokkos"] = "disabled";
    }
#else
     n["runtimes/ascent/vtkm/status"] = "disabled";
#endif

#if defined(ASCENT_MFEM_ENABLED)
    n["runtimes/ascent/mfem/status"] = "enabled";
#else
    n["runtimes/ascent/mfem/status"] = "disabled";
#endif

#if defined(ASCENT_HDF5_ENABLED)
    n["runtimes/ascent/hdf5/status"] = "enabled";
#else
    n["runtimes/ascent/hdf5/status"] = "disabled";
#endif


#if defined(ASCENT_ADIOS2_ENABLED)
    n["runtimes/ascent/adios2/status"] = "enabled";
#else
    n["runtimes/ascent/adios2/status"] = "disabled";
#endif

#if defined(ASCENT_FIDES_ENABLED)
    n["runtimes/ascent/fides/status"] = "enabled";
#else
    n["runtimes/ascent/fides/status"] = "disabled";
#endif

#if defined(ASCENT_GENTEN_ENABLED)
    n["runtimes/ascent/genten/status"] = "enabled";
#else
    n["runtimes/ascent/genten/status"] = "disabled";
#endif

#if defined(ASCENT_BABELFLOW_ENABLED)
    n["runtimes/ascent/babelflow/status"] = "enabled";
#else
    n["runtimes/ascent/babelflow/status"] = "disabled";
#endif

#if defined(ASCENT_WEBSERVER_ENABLED)
    n["runtimes/ascent/webserver/status"] = "enabled";
#else
    n["runtimes/ascent/webserver/status"] = "disabled";
#endif

    n["runtimes/flow/status"] = "enabled";

    n["default_runtime"] = "ascent";

}


//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------


