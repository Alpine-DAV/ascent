//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_steering_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_steering_filters.hpp"

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
#include <algorithm>


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
Steering::Steering()
:Filter()
{
    // Commands
    commands["exit"] = [this]()
    { this->exit_shell(); };
    commands["help"] = [this]()
    { this->print_help(); };
    commands["list"] = [this]()
    { this->list_callbacks(); };
    commands["param"] = [this]()
    { this->print_params(); };
    commands["run"] = [this]()
    { this->empty_run(); };

    // Descriptions
    descriptions["exit"] = "Exit the shell.";
    descriptions["help"] = "Print this help message.";
    descriptions["list"] = "List all registered Ascent callbacks.";
    descriptions["param"] = "Prints the current parameters. Can set, modify, or parameters with 'param add|edit|remove key value.";
    descriptions["run"] = "Run an Ascent callback with 'run my_callback_name'.";

#ifdef ASCENT_MPI_ENABLED
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
#else
    m_rank = 0;
#endif
}

//-----------------------------------------------------------------------------
Steering::~Steering()
{
// empty
}

//-----------------------------------------------------------------------------
void
Steering::declare_interface(Node &i)
{
    i["type_name"] = "steering";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
Steering::verify_params(const conduit::Node &params,
                       conduit::Node &info)
{
    info.reset();
    return true;
}

//-----------------------------------------------------------------------------
void
Steering::execute()
{
    running = true;
    empty_params = true;

    if (m_rank == 0)
    {
        std::cout << "-------------Entering interactive steering mode-------------" << std::endl;
        std::cout << "Type 'help' to see available commands" << std::endl << std::endl;
    }

    std::string input;
    int input_size = 0;
    while (running)
    {
        
        if (m_rank == 0)
        {
            std::cout << "steering>" << std::endl;
            std::getline(std::cin, input);
            input_size = input.size();
        }

#ifdef ASCENT_MPI_ENABLED
        MPI_Bcast(&input_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (m_rank > 0)
        {
            input.resize(input_size);
        }
        MPI_Bcast(const_cast<char *>(input.data()), input_size, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

        // Tokenize user input
        std::istringstream iss(input);
        std::string cmd;
        iss >> cmd;
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token)
        {
            tokens.push_back(token);
        }
        parse_input(cmd, tokens);
    }
    if (m_rank == 0)
    {
        std::cout << std::endl << "-------------Exiting interactive steering mode-------------" << std::endl;
    }
}

void Steering::empty_run()
{
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Error]" << std::endl;
        std::cout << "Must specify a callback to run, for example: 'run my_callback_name'" << std::endl << std::endl;
    }
}

void Steering::exit_shell()
{
    running = false;
}

void Steering::list_callbacks()
{
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Available Callbacks]" << std::endl;
        std::vector<std::string> void_callback_names = ascent::get_void_callbacks();
        if (void_callback_names.size() == 0)
        {
            std::cout << "void: no callbacks registered";
        }
        else
        {
            std::cout << "void: ";
            for (const auto &callback : void_callback_names)
            {
                std::cout << callback << " ";
            }
        }
        std::cout << std::endl;
        std::vector<std::string> bool_callback_names = ascent::get_bool_callbacks();
        if (bool_callback_names.size() == 0)
        {
            std::cout << "bool: no callbacks registered";
        }
        else
        {
            std::cout << "bool: ";
            for (const auto &callback : bool_callback_names)
            {
                std::cout << callback << " ";
            }
        }
        std::cout << std::endl << std::endl;
    }
}

void Steering::modify_params(std::vector<std::string> args)
{
    std::string cmd = args.size() > 0 ? args[0] : "";
    std::string arg = args.size() > 1 ? args[1] : "";

    if (cmd == "reset")
    {
        params.reset();
        empty_params = true;
    }
    else if (cmd == "delete" && !arg.empty())
    {
        params[arg].reset();
    }
    else if (!cmd.empty() && !arg.empty())
    {
        bool assigned = false;

        // Is the input numeric?
        if (!assigned)
        {
            try
            {
                double possible_number = std::stod(arg);
                params[cmd] = possible_number;
                assigned = true;
                empty_params = false;
            }
            catch (const std::invalid_argument&)
            {
            }
            catch (const std::out_of_range&)
            {
            }
        }

        // It wasn't numeric, treat it as a string
        if (!assigned)
        {
            params[cmd] = arg;
            empty_params = false;
        }
    }

    print_params();
}

void Steering::print_help()
{
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Available Commands]" << std::endl;
        for (const auto &cmd : descriptions)
        {
            std::cout << cmd.first << "\t-   " << cmd.second << std::endl;
        }
        std::cout << std::endl;
    }
}

void Steering::print_params()
{
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Current Params]";
        if (empty_params)
        {
            std::cout << std::endl << "None" << std::endl << std::endl;
        }
        else
        {
            params.print();
        }
    }
}

void Steering::run_callback(std::vector<std::string> args)
{
    try
    {
        std::string callback = args[0];

        bool has_callback = false;
        std::vector<std::string> void_callback_names = ascent::get_void_callbacks();
        auto it = std::find(void_callback_names.begin(), void_callback_names.end(), callback);

        if (it != void_callback_names.end()) {
            has_callback = true;
            if (m_rank == 0)
            {
                std::cout << std::endl << "Running callback: " << callback << "\n" << std::endl;
            }
        }
        else
        {
            if (m_rank == 0)
            {
                std::cout << std::endl << "[Error]" << std::endl;
                std::cout << "There is no callback named '" << callback << "'" << std::endl << std::endl;
            }
        }

        if (!has_callback)
        {
            return;
        }

        ascent::execute_callback(callback, params, output);

        // This might not be necessary?
#ifdef ASCENT_MPI_ENABLED
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        
        if (m_rank == 0)
        {
            std::cout << std::endl << "[Output]";
            output.print();
            std::cout << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}

void Steering::parse_input(std::string cmd, std::vector<std::string> args)
{
    if (commands.find(cmd) == commands.end())
    {
        if (m_rank == 0)
        {
            std::cout << "[Error]" << std::endl;
            std::cout << std::endl << "Unknown command: " << cmd << std::endl;
            print_help();
        }
        return;
    }

    if (cmd == "run" && !args.empty())
    {
        run_callback(args);
    }
    else if (cmd == "param" && !args.empty())
    {
        modify_params(args);
    }
    else
    {
        commands[cmd]();
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
