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

using namespace std;
using namespace conduit;
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
// empty
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
    bool res = true;
    // This optional parameter exists for automated testing purposes
    if(params.has_child("explicit_command"))
    {
        if(!params["explicit_command"].dtype().is_string())
        {
            info["errors"].append() = "optional entry 'explicit_command' must"
                                      " be a string";
            res = false;
        }
    }
    return res;
}

//-----------------------------------------------------------------------------
void
Steering::execute()
{
    // Map of commands to function pointers
    m_commands["exit"] = [this](){this->exit_shell();};
    m_commands["help"] = [this](){this->print_help();};
    m_commands["list"] = [this](){this->list_callbacks();};
    m_commands["param"] = [this](){this->print_params();};
    m_commands["run"] = [this](){this->empty_run();};

    // Descriptions for each command
    m_descriptions["exit"] = "Exit the steering interface.";
    m_descriptions["help"] = "Print this help message.";
    m_descriptions["list"] = "List all registered Ascent callbacks.";
    m_descriptions["param"] = "Modify current params.\n\t\
        - Print:   param\n\t\
        - Add:     param <key> <value>\n\t\
        - Delete:  param delete <key>\n\t\
        - Reset:   param reset";
    m_descriptions["run"] = "Run an Ascent callback.\n\t\
        - Example: run <callback_name>";

#ifdef ASCENT_MPI_ENABLED
    // Grab the MPI communicator we're supposed to use
    m_mpi_comm = MPI_Comm_f2c(flow::Workspace::default_mpi_comm());
    MPI_Comm_rank(m_mpi_comm, &m_rank);
    m_params["mpi_comm"] = MPI_Comm_c2f(m_mpi_comm);
#else
    m_rank = 0;
#endif
    m_params["mpi_rank"] = m_rank;

    // Print header
    if (m_rank == 0)
    {
        std::cout << std::endl << "-------------Entering interactive";
        std::cout << " steering mode for Ascent--------------" << std::endl;
        std::cout << "Type 'help' to see available commands" << std::endl;
        std::cout << std::endl;
    }

    // Setup variables
    m_running = true;
    while (m_running)
    {
        std::string input = "";
        int input_size = input.size();

        if(params().has_path("explicit_command"))
        {
            // This path exists for automated testing purposes
            input = params()["explicit_command"].as_string();
            if (m_rank == 0)
            {
                std::cout << "steering>" << std::endl;
                std::cout << "(explicit_command): " << input << std::endl;
            }
            input_size = input.size();
            m_running = false;
        }
        else if (m_rank == 0)
        {
            // Blocks forever waiting for user input
            // TODO: it would be cool to eventually implement a timeout here
            std::cout << "steering>" << std::endl;
            std::getline(std::cin, input);
            input_size = input.size();
        }

#ifdef ASCENT_MPI_ENABLED
        // The input string gets broadcast, parsed, and executed by all MPI
        // ranks. This enables us to 1) maintain param parity and 2)
        // execute callbacks that make use of MPI.
        MPI_Bcast(&input_size, 1, MPI_INT, 0, m_mpi_comm);
        if (m_rank > 0)
        {
            input.resize(input_size);
        }
        MPI_Bcast(const_cast<char *>(input.data()),
                  input_size,
                  MPI_CHAR,
                  0,
                  m_mpi_comm);
#endif

        // Parse and execute user input. Users can't naturally enter
        // multi-line commands, but we support it here for automated
        // testing purposes
        std::istringstream iss(input);
        std::string line;
        while (std::getline(iss, line))
        {
            std::istringstream lineStream(line);
            std::string cmd;
            lineStream >> cmd;
        
            std::vector<std::string> tokens;
            std::string token;
            while (lineStream >> token)
            {
                tokens.push_back(token);
            }

            parse_input(cmd, tokens);
        }
    }

    // Print footer
    if (m_rank == 0)
    {
        std::cout << "-------------Exiting interactive steering mode for";
        std::cout << " Ascent-------------" << std::endl << std::endl;
    }
}

//-----------------------------------------------------------------------------
void
Steering::empty_run()
{
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Error]" << std::endl;
        std::cout << "Did not specify a callback to run. Example usage: run";
        std::cout << " <callback_name>" << std::endl << std::endl;
    }
}

//-----------------------------------------------------------------------------
void
Steering::exit_shell()
{
    if (m_rank == 0)
    {
        std::cout << std::endl;
    }
    m_running = false;
}

//-----------------------------------------------------------------------------
void
Steering::list_callbacks()
{
    // Let users see which callbacks were registered with Ascent
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Available Callbacks]" << std::endl;
        std::vector<std::string> void_callback_names;
        ascent::get_void_callbacks(void_callback_names);
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
        std::vector<std::string> bool_callback_names;
        ascent::get_bool_callbacks(bool_callback_names);
        if (bool_callback_names.size() == 0)
        {
            std::cout << "bool: no callbacks registered";
        }
        else
        {
            // We currently don't let users execute bool callbacks via this
            // interface, but we may as well list them here for debug
            // purposes
            std::cout << "bool: ";
            for (const auto &callback : bool_callback_names)
            {
                std::cout << callback << " ";
            }
        }
        std::cout << std::endl << std::endl;
    }
}

//-----------------------------------------------------------------------------
void
Steering::print_help()
{
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Available Commands]" << std::endl;
        for (const auto &cmd : m_descriptions)
        {
            std::cout << cmd.first << "\t-   " << cmd.second << std::endl;
        }
        std::cout << std::endl;
    }
}

//-----------------------------------------------------------------------------
void
Steering::print_params()
{
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Current Params]";
        if (m_params.number_of_children() > 0)
        {
            m_params.print();
        }
        else
        {
            std::cout << std::endl << "None" << std::endl << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
void
Steering::run_callback(const std::string &callback_name)
{
    bool has_callback = false;

    // Void callback name iterator
    std::vector<std::string> void_callback_names;
    ascent::get_void_callbacks(void_callback_names);
    auto void_it = std::find(void_callback_names.begin(),
                             void_callback_names.end(),
                             callback_name);

    // Bool callback name iterator
    std::vector<std::string> bool_callback_names;
    ascent::get_bool_callbacks(bool_callback_names);
    auto bool_it = std::find(bool_callback_names.begin(),
                             bool_callback_names.end(),
                             callback_name);

    // Make sure that the callback actually exists before executing it. This
    // lets us print a friendlier error in case it doesn't
    if (void_it != void_callback_names.end())
    {
        has_callback = true;
        if (m_rank == 0)
        {
            std::cout << std::endl << "Running callback: " << callback_name;
            std::cout << std::endl;
        }
    }
    else if (bool_it != bool_callback_names.end() && m_rank == 0)
    {
        // TODO: would it make sense to let users execute bool callbacks?
        std::cout << std::endl << "[Error]" << std::endl;
        std::cout << "Directly executing bool callbacks is not";
        std::cout << " supported from this interface." << std::endl;
        std::cout << std::endl;
    }
    else if (m_rank == 0)
    {
        std::cout << std::endl << "[Error]" << std::endl;
        std::cout << "There is no registered callback named '";
        std::cout << callback_name << "'" << std::endl << std::endl;
    }

    if (!has_callback)
    {
        return;
    }

    // Reset any output from a previous callback
    m_output = conduit::Node();

    // It's not a great experience if the interface crashes due to an exception
    // getting thrown within a callback
    try
    {
        ascent::execute_callback(callback_name, m_params, m_output);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    // TODO: reevaluate if an MPI barrier is necessary here
#ifdef ASCENT_MPI_ENABLED
    MPI_Barrier(m_mpi_comm);
#endif
    
    if (m_rank == 0)
    {
        std::cout << std::endl << "[Output]";
        if (m_output.number_of_children() > 0)
        {
            m_output.print();
        }
        else
        {
            std::cout << std::endl << "None" << std::endl << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------
void
Steering::modify_params(const std::vector<std::string> &tokens)
{
    std::string cmd = tokens.size() > 0 ? tokens[0] : "";
    std::string arg = tokens.size() > 1 ? tokens[1] : "";

    if (cmd == "reset")
    {
        // Start fresh with a new Conduit node
        m_params = conduit::Node();
#ifdef ASCENT_MPI_ENABLED
        m_params["mpi_comm"] = MPI_Comm_c2f(m_mpi_comm);
#endif
        m_params["mpi_rank"] = m_rank; 
    }
    else if (cmd == "delete" && !arg.empty())
    {
        // Remove a specific param
        if(arg != "mpi_comm" && arg != "mpi_rank" && m_params.has_child(arg))
        {
            m_params.remove(arg);
        }
    }
    else if (!cmd.empty() && !arg.empty())
    {
        bool assigned = false;

        // Is the input numeric?
        if (!assigned)
        {
            try
            {
                // The side effect of this is that integers are also 
                double possible_number = std::stod(arg);
                m_params[cmd] = possible_number;
                assigned = true;
            }
            catch (const std::invalid_argument&)
            {
            }
            catch (const std::out_of_range&)
            {
            }
        }

        // The input wasn't numeric, so we treat it as a string
        if (!assigned)
        {
            m_params[cmd] = arg;
        }
    }

    print_params();
}

//-----------------------------------------------------------------------------
void
Steering::parse_input(const std::string &cmd,
                      const std::vector<std::string> &args)
{
    if (m_commands.find(cmd) == m_commands.end())
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
        std::string callback_name = args[0];
        run_callback(callback_name);
    }
    else if (cmd == "param" && !args.empty())
    {
        modify_params(args);
    }
    else
    {
        m_commands[cmd]();
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
