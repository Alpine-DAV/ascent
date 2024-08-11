//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_steering_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_STEERING_FILTERS
#define ASCENT_RUNTIME_STEERING_FILTERS

#include <ascent.hpp>
#include <flow_filter.hpp>

#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

// std includes
#include <algorithm>
#include <functional>

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
///
/// Filters Related to Blueprint
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class ASCENT_API Steering : public ::flow::Filter
{
public:
    Steering();
    ~Steering();
    virtual void declare_interface(conduit::Node &i);
    virtual bool verify_params(const conduit::Node &params,
                                     conduit::Node &info);
    virtual void execute();
private:
    std::map<std::string, std::function<void()>> m_commands;
    std::map<std::string, std::string> m_descriptions;
    conduit::Node m_params;
    conduit::Node m_output;
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm m_mpi_comm;
#endif
    int m_rank;
    bool m_running;

    void empty_run();
    void exit_shell();
    void list_callbacks();
    void print_help();
    void print_params();
    void run_callback(const std::string &callback_name);
    void modify_params(const std::vector<std::string> &tokens);
    void parse_input(const std::string &cmd,
                     const std::vector<std::string> &args);
};

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

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------
