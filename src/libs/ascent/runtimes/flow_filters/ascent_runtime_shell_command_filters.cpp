//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_shell_command_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_shell_command_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

#include <mpi.h>

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_expression_eval.hpp>
#include <ascent_data_object.hpp>
#include <ascent_logging.hpp>
#include <ascent_runtime_param_check.hpp>

#include <flow_graph.hpp>
#include <flow_workspace.hpp>

using namespace conduit;
using namespace std;

using namespace flow;

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
ShellCommand::ShellCommand()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
ShellCommand::~ShellCommand()
{
// empty
}

//-----------------------------------------------------------------------------
void
ShellCommand::declare_interface(Node &i)
{
    i["type_name"]   = "shell_command";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
ShellCommand::verify_params(const conduit::Node &params,
                            conduit::Node &info)
{
    info.reset();
    bool res = check_string("command", params, info, true);

    std::vector<std::string> valid_paths;
    valid_paths.push_back("command");
    valid_paths.push_back("mpi_behavior");

    std::vector<std::string> ignore_paths;
    // don't go down the actions path
    ignore_paths.push_back("actions");

    std::string surprises = surprise_check(valid_paths, ignore_paths,params);

    if(surprises != "")
    {
      res = false;
      info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
ShellCommand::execute()
{
    if(!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("Shell command input must be a data object");
    }

    std::string cmd = params()["command"].as_string();

    #ifdef ASCENT_MPI_ENABLED
        std::string mpi_behavior = params()["mpi_behavior"].as_string();
        if (mpi_behavior == "root") {
            int comm = Workspace::default_mpi_comm();
            int rank;
            MPI_Comm_rank(MPI_Comm_f2c(comm), &rank);
            if (rank == 0) {
                system(cmd.c_str());
            }
            return;
        }
    #endif

    system(cmd.c_str());
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





