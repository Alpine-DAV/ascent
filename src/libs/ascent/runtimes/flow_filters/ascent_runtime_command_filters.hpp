//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_runtime_command_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_RUNTIME_COMMAND_FILTERS
#define ASCENT_RUNTIME_COMMAND_FILTERS

#include <ascent.hpp>

#include <flow_filter.hpp>

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
class ASCENT_API Command : public ::flow::Filter
{
public:
    Command();
    ~Command();
    virtual void declare_interface(conduit::Node &i);
    virtual bool verify_params(const conduit::Node &params,
                               conduit::Node &info);
    virtual void execute();

    void static  register_callback(const std::string &callback_name,
                                   void (*callback_function)(conduit::Node &, conduit::Node &));
    void static  register_callback(const std::string &callback_name,
                                   bool (*callback_function)(void));
    void static  execute_command_list(const std::vector<std::string> commands,
                                      const std::string &command_type);
    void static  execute_shell_command(std::string command);
    void static  execute_void_callback(std::string callback_name,
                                       conduit::Node &params,
                                       conduit::Node &output);
    bool static  execute_bool_callback(std::string callback_name);

    static std::map<std::string, void (*)(conduit::Node &, conduit::Node &)> m_void_callback_map;
    static std::map<std::string, bool (*)(void)> m_bool_callback_map;
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
