//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_flow_runtime.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_FLOW_RUNTIME_HPP
#define ASCENT_FLOW_RUNTIME_HPP

#include <ascent.hpp>
#include <ascent_runtime.hpp>

#include <ascent_web_interface.hpp>

#include <flow.hpp>




//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

class ASCENT_API FlowRuntime : public Runtime
{
public:

    // Creation and Destruction
    FlowRuntime();
    virtual ~FlowRuntime();

    // Main runtime interface methods used by the ascent interface.
    void  Initialize(const conduit::Node &options) override;

    void  RegisterCallback(const std::string &callback_name,
                           void (*callback_function)(conduit::Node &)) override;
    void  RegisterCallback(const std::string &callback_name,
                           bool (*callback_function)(void)) override;
    void  Publish(const conduit::Node &data)    override;
    void  Execute(const conduit::Node &actions) override;

    void                 Info(conduit::Node &out) override;
    conduit::Node       &Info() override;

    void  Cleanup() override;

private:
    // holds options passed to initialize
    conduit::Node     m_runtime_options;
    // conduit node that (externally) holds the data from the simulation
    conduit::Node     m_data;

    conduit::Node     m_info;

    flow::Workspace w;

    void              ResetInfo();
    void              ConnectSource();

    WebInterface      m_web_interface;


};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


