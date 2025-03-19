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
#include <ascent_logging.hpp>
#include <ascent_logging_old.hpp>
#include <runtimes/ascent_main_runtime.hpp>
#include <utils/ascent_string_utils.hpp>
#include <flow.hpp>

#include <conduit_fmt/conduit_fmt.h>

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
namespace detail
{

//-----------------------------------------------------------------------------
bool
check_for_file(const std::string &file_name,
               int mpi_comm_id)
{
    int rank = 0;
#ifdef ASCENT_MPI_ENABLED
    if(mpi_comm_id == -1)
    {
        // nothing we can do
        return false;
    }

    MPI_Comm mpi_comm = MPI_Comm_f2c(mpi_comm_id);
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

    return has_file == 1;

}

//-----------------------------------------------------------------------------
void
load_included_files_in_node_tree(conduit::Node &node, int mpi_comm_id)
{
    // This function recursively traverses the node tree searching for include statements
    // If one is found, the node attempts to load the file's contents and add it to the node tree
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

    if(node.has_child("include"))
    {
        int include_file_valid = 0;
        std::string emsg = "";
        std::string file_name = "";
        
        // Only want to update the node on rank 0
        if (rank == 0) {
            file_name = node.fetch("include").as_string();
            node.remove_child("include");

            // Determine file protocol from file extension
            std::string curr,next;
            std::string protocol = "json";
            conduit::utils::rsplit_string(file_name,
                                            ".",
                                            curr,
                                            next);
            if(curr == "yaml")
            {
                protocol = "yaml";
            }

            // Try loading in the included path
            try
            {
                conduit::Node file_node;
                file_node.load(file_name, protocol);
                node.update(file_node);
                include_file_valid = 1;
            }
            catch(conduit::Error &e)
            {
                include_file_valid = 0;
                emsg = e.message();
            }
        }

#ifdef ASCENT_MPI_ENABLED
        // make sure all ranks error if the parsing on rank 0 failed.
        MPI_Bcast(&include_file_valid, 1, MPI_INT, 0, mpi_comm);

        // Pass the error to all ranks so the error message matches
        conduit::Node n_emsg;
        if(rank == 0)
        {
        n_emsg.set(emsg);
        }

        conduit::relay::mpi::broadcast_using_schema(n_emsg, 0, mpi_comm);
        emsg = n_emsg.as_string();
#endif

        if(include_file_valid == 0)
        {
            // Raise Error
            ASCENT_ERROR("Failed to load actions file: " << file_name
                        << "\n" << emsg);
        }

#ifdef ASCENT_MPI_ENABLED
        // If successful, make sure that all ranks received the updated node
        relay::mpi::broadcast_using_schema(node, 0, mpi_comm);
#endif
    }

    // If there are any children, recurse over the children
    // This includes any newly included children
    for (index_t i = 0; i<node.number_of_children(); i++)
    {
        load_included_files_in_node_tree(node[i], mpi_comm_id);
    }
}

//-----------------------------------------------------------------------------
int
ParRank(int comm_id)
{
    int rank = 0;

#if defined(ASCENT_MPI_ENABLED)
    if(comm_id == -1)
    {
      // do nothing, an error will be thrown later
      // so we can respect the exception handling
      return 0;
    }
    MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
    MPI_Comm_rank(mpi_comm, &rank);
#endif

    return rank;
}

//-----------------------------------------------------------------------------
int
ParSize(int comm_id)
{
int comm_size=1;

#if defined(ASCENT_MPI_ENABLED)
    if(comm_id == -1)
    {
      // do nothing, an error will be thrown later
      // so we can respect the exception handling
      return 1;
    }
    MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
    MPI_Comm_size(mpi_comm, &comm_size);
#endif

    return comm_size;
}

//-----------------------------------------------------------------------------
void
quiet_handler(const std::string &,
              const std::string &,
              int )
{
}

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

    detail::load_included_files_in_node_tree(node, mpi_comm_id);
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

        int par_rank = detail::ParRank(comm_id);
        int par_size = detail::ParSize(comm_id);

        detail::load_included_files_in_node_tree(processed_opts, comm_id);

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

        Node echo_opts;
        echo_opts["echo_threshold"] = "info";
        echo_opts["ranks"] = "root";

        // messages echoed to std out
        if(m_options.has_path("messages"))
        {
            if(m_options["messages"].dtype().is_string())
            {
                std::string msgs_opt = m_options["messages"].as_string();
                if( msgs_opt == "verbose")
                {
                    m_verbose_msgs = true;
                    echo_opts["echo_threshold"] = "all";
                }
                else if(msgs_opt == "quiet")
                {
                    m_verbose_msgs = false;
                    echo_opts["echo_threshold"] = "error";
                }
            }
            else if(m_options["messages"].dtype().is_object())
            {
                const Node &msg_ops = m_options["messages"];
                echo_opts.update(msg_ops);
                if(msg_ops.has_child("echo_threshold") &&
                   msg_ops["echo_threshold"].dtype().is_string() ) 
                {
                    std::string echo_thresh = msg_ops["echo_threshold"].as_string();
                    if(echo_thresh == "none" || 
                       echo_thresh == "error" ||
                       echo_thresh == "warn" )
                    {
                        m_verbose_msgs = false;
                    }
                    else
                    {
                        m_verbose_msgs = true;
                    }
                }
            }
        }
        
        ascent::Logger &logger = ascent::Logger::instance();

        // setup echo
        logger.set_echo_threshold(echo_opts["echo_threshold"].as_string());

        // controls for mpi ranks
        // if ranks == "root"
        //   rank 0 is what is specified, echo_threshold = "none" for all others
        // if ranks == "all"
        //   echo_threshold = specified option used for all ranks (alreay handled above)
        // if ranks == [list of ints] (also accepts single int)
        //   echo_threshold = specified option used for ranks in the list

        if(echo_opts["ranks"].dtype().is_number()) // list of ints case
        {
            int64_accessor ranks_list = echo_opts["ranks"].value();
            bool active = false;
            for(index_t i=0; i < ranks_list.number_of_elements(); i++)
            {
                if(par_rank == ranks_list[i] )
                {
                    active = true;
                }
            }

            if(!active)
            {
                logger.set_echo_threshold("none");
            }
        }
        else // string options case
        {
            std::string log_ranks_str = echo_opts["ranks"].as_string();

            if(log_ranks_str == "root")
            {
                if(par_rank != 0)
                {
                    logger.set_echo_threshold("none");
                }
            }
        }

        // logging options
        //
        //   logging: true
        //
        //   logging:
        //      file_pattern: zzzz
        //      log_threshold:  info
        //

        Node logging_opts;
        logging_opts["enabled"] = 0;
        logging_opts["ranks"]   = "all";
#if defined(ASCENT_MPI_ENABLED)
        logging_opts["file_pattern"]  = "ascent_log_output_rank_{rank:05d}.yaml";
#else
        logging_opts["file_pattern"]  = "ascent_log_output.yaml";
#endif
        logging_opts["log_threshold"] = "debug";

        if(m_options.has_path("logging"))
        {
            if(m_options["logging"].dtype().is_string() &&
               m_options["logging"].as_string() == "true")
            {
                logging_opts["enabled"] = 1;
            }
            else if(m_options["logging"].dtype().is_object())
            {
                logging_opts["enabled"] = 1;
                // pull over options
                logging_opts.update(m_options["logging"]);
            }
        }

        // controls for mpi ranks
        // if ranks == "root"
        //   open log on rank 0, do not open on all others
        // if ranks == "all"
        //   open log on all ranks
        // if ranks == [list of ints] (also accepts single int)
        //   open log on ranks specified in the list

        if(logging_opts["ranks"].dtype().is_number()) // list of ints case
        {
            int64_accessor ranks_list = logging_opts["ranks"].value();
            bool active = false;
            for(index_t i=0; i < ranks_list.number_of_elements(); i++)
            {
                if(par_rank == ranks_list[i] )
                {
                    active = true;
                }
            }

            if(!active)
            {
                logging_opts["enabled"] = 0;
            }
        }
        else // string options case
        {
            std::string log_ranks_str = logging_opts["ranks"].as_string();

            if(log_ranks_str == "root")
            {
                if(par_rank != 0)
                {
                    logging_opts["enabled"] = 0;
                }
            }
            // all already supported if logging is enabled
        }


        if(logging_opts["enabled"].to_int() == 1)
        {
            logger.set_log_threshold(logging_opts["log_threshold"].as_string());
            std::string file_pattern = logging_opts["file_pattern"].as_string();
        #if defined(ASCENT_MPI_ENABLED)
            ASCENT_LOG_OPEN_RANK( file_pattern, par_rank ) // mpi par
            ASCENT_LOG_DEBUG(conduit_fmt::format("mpi info: rank={}, size={}",
                                                  par_rank,
                                                  par_size));
        #else
            ASCENT_LOG_OPEN( file_pattern ) // serial
            ASCENT_LOG_DEBUG("mpi not enabled");
        #endif
        }


        // exception controls
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
            conduit::utils::set_info_handler(detail::quiet_handler);
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
            int mpi_comm_id = m_options["mpi_comm"].to_int();

            Node processed_actions(actions);
            detail::load_included_files_in_node_tree(processed_actions, mpi_comm_id);

            if(m_actions_file == "<<UNSET>>")
            {
                m_actions_file = "ascent_actions.json";

                if(!detail::check_for_file(m_actions_file, mpi_comm_id))
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
                if(!detail::check_for_file(m_actions_file, mpi_comm_id))
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
                                 mpi_comm_id);

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
         ASCENT_LOG_CLOSE();
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
        ASCENT_LOG_CLOSE();
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
    ASCENT_LOG_DEBUG(msg);
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
    ASCENT_LOG_DEBUG(msg + " " + details);
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

// Callback maps
static std::map<std::string, void (*)(conduit::Node &, conduit::Node &)> m_void_callback_map;
static std::map<std::string, bool (*)(void)> m_bool_callback_map;

//-----------------------------------------------------------------------------
void
register_callback(const std::string &callback_name,
                  void (*callback_function)
                  (conduit::Node &, conduit::Node &))
{
    if (callback_name == "")
    {
        ASCENT_ERROR("cannot register an anonymous void callback");
    }
    else if (m_void_callback_map.count(callback_name) != 0)
    {
        ASCENT_ERROR("cannot register more than one void callback under the name '" << callback_name << "'");
    }
    else if (m_bool_callback_map.count(callback_name) != 0)
    {
        ASCENT_ERROR("cannot register both a void and bool callback under the same name '" << callback_name << "'");
    }
    m_void_callback_map.insert(std::make_pair(callback_name, callback_function));
}

//-----------------------------------------------------------------------------
void
register_callback(const std::string &callback_name,
                  bool (*callback_function)(void))
{
    if (callback_name == "")
    {
        ASCENT_ERROR("cannot register an anonymous bool callback");
    }
    else if (m_bool_callback_map.count(callback_name) != 0)
    {
        ASCENT_ERROR("cannot register more than one bool callback under the name '" << callback_name << "'");
    }
    else if (m_void_callback_map.count(callback_name) != 0)
    {
        ASCENT_ERROR("cannot register both a void and bool callback under the same name '" << callback_name << "'");
    }
    m_bool_callback_map.insert(std::make_pair(callback_name, callback_function));
}

//-----------------------------------------------------------------------------
void
execute_callback(const std::string &callback_name,
                 conduit::Node &params,
                 conduit::Node &output)
{
    if (m_void_callback_map.count(callback_name) != 1)
    {
        ASCENT_ERROR("requested void callback '" << callback_name << "' was never registered");
    }
    auto callback_function = m_void_callback_map.at(callback_name);
    return callback_function(params, output);
}

//-----------------------------------------------------------------------------
bool
execute_callback(const std::string &callback_name)
{
    if (m_bool_callback_map.count(callback_name) != 1)
    {
        ASCENT_ERROR("requested bool callback '" << callback_name << "' was never registered");
    }
    auto callback_function = m_bool_callback_map.at(callback_name);
    return callback_function();
}

//-----------------------------------------------------------------------------
void
get_void_callbacks(std::vector<std::string> &callback_names)
{
    for (const auto &pair : m_void_callback_map)
    {
        callback_names.push_back(pair.first);
    }
}

//-----------------------------------------------------------------------------
void
get_bool_callbacks(std::vector<std::string> &callback_names)
{
    for (const auto &pair : m_bool_callback_map)
    {
        callback_names.push_back(pair.first);
    }
}

//-----------------------------------------------------------------------------
void
reset_callbacks()
{
    m_void_callback_map.clear();
    m_bool_callback_map.clear();
}

//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------


