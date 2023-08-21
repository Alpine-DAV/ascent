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

std::map<std::string, void (*)(conduit::Node &, conduit::Node &)> Command::m_void_callback_map;
std::map<std::string, bool (*)(void)> Command::m_bool_callback_map;

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
Command::register_callback(const std::string &callback_name,
                           void (*callback_function)(conduit::Node &, conduit::Node &))
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
Command::register_callback(const std::string &callback_name,
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
Command::execute_command_list(const std::vector<std::string> commands,
                              const std::string &command_type)
{
    if (command_type == "callback")
    {
        conduit::Node params;
        conduit::Node output;
        for (int i = 0; i < commands.size(); i++)
        {
            execute_void_callback(commands.at(i), params, output);
        }
    } else if (command_type == "shell_command")
    {
        for (int i = 0; i < commands.size(); i++)
        {
            execute_shell_command(commands.at(i));
        }
    }
}

//-----------------------------------------------------------------------------
void
Command::execute_shell_command(std::string command)
{
    system(command.c_str());
}

//-----------------------------------------------------------------------------
void
Command::execute_void_callback(std::string callback_name, conduit::Node &params, conduit::Node &output)
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
Command::execute_bool_callback(std::string callback_name)
{
    if (m_bool_callback_map.count(callback_name) != 1)
    {
        ASCENT_ERROR("requested bool callback '" << callback_name << "' was never registered");
    }
    auto callback_function = m_bool_callback_map.at(callback_name);
    return callback_function();
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
