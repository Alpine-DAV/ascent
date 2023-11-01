//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_command_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_command_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_data_object.hpp>
#include <ascent_expression_eval.hpp>
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
Command::Command()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Command::~Command()
{
// empty
}

//-----------------------------------------------------------------------------
void
Command::declare_interface(Node &i)
{
    i["type_name"] = "command";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
Command::verify_params(const conduit::Node &params,
                       conduit::Node &info)
{
    info.reset();

    bool has_callback = params.has_path("callback");
    bool has_shell_command = params.has_path("shell_command");

    bool res = false;
    if (!has_callback && !has_shell_command)
    {
        info["errors"].append() = "There was no callback or shell command defined";
    }
    else if (has_callback && has_shell_command)
    {
        info["errors"].append() = "Both a callback and shell command are "
                                  "present. Choose one or the other.";
    }
    else if(has_callback && !params["callback"].dtype().is_string())
    {
        info["errors"].append() = "Callbacks must be a string";  
    }
    else if(has_shell_command && !params["shell_command"].dtype().is_string())
    {
        info["errors"].append() = "Shell commands must be a string";  
    }
    else
    {
        res = true;
    }

    std::vector<std::string> valid_paths;
    valid_paths.push_back("callback");
    valid_paths.push_back("shell_command");
    valid_paths.push_back("mpi_behavior");

    std::vector<std::string> ignore_paths;

    std::string surprises = surprise_check(valid_paths, ignore_paths, params);

    if (surprises != "")
    {
        res = false;
        info["errors"].append() = surprises;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
Command::execute()
{

    if (!input(0).check_type<DataObject>())
    {
        ASCENT_ERROR("Command input must be a data object");
    }

    bool has_callback = params().has_path("callback");
    std::string command_type = has_callback ? "callback" : "shell_command";

    std::stringstream ss(params()[command_type].as_string());

    std::vector<std::string> commands;
    std::string command;
    while(std::getline(ss, command, '\n'))
    {
        commands.push_back(command);
    }

    #ifdef ASCENT_MPI_ENABLED
    bool has_mpi_behavior = params().has_path("mpi_behavior");
    if (has_mpi_behavior)
    {
        std::string mpi_behavior = params()["mpi_behavior"].as_string();
        if (mpi_behavior == "root")
        {
            int comm = Workspace::default_mpi_comm();
            int rank;
            MPI_Comm_rank(MPI_Comm_f2c(comm), &rank);
            if (rank == 0)
            {
                execute_command_list(commands, command_type);
            }
            return;
        }
    }
    #endif

    execute_command_list(commands, command_type);
}

//-----------------------------------------------------------------------------
void
Command::execute_command_list(const std::vector<std::string> commands,
                              const std::string &command_type)
{
    if (command_type == "callback")
    {
        conduit::Node params;
        conduit::Node output;
        for (int i = 0; i < commands.size(); i++)
        {
            ascent::execute_callback(commands.at(i), params, output);
        }
    } else if (command_type == "shell_command")
    {
        for (int i = 0; i < commands.size(); i++)
        {
            system(commands.at(i).c_str());
        }
    }
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
