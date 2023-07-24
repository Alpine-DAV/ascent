//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_flow_runtime.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_flow_runtime.hpp"

// standard lib includes
#include <string.h>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit_blueprint.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

#include <flow.hpp>
#include <ascent_runtime_filters.hpp>

using namespace conduit;
using namespace std;


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
FlowRuntime::FlowRuntime()
:Runtime()
{
    flow::filters::register_builtin();
    ResetInfo();
}

//-----------------------------------------------------------------------------
FlowRuntime::~FlowRuntime()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main runtime interface methods called by the ascent interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
FlowRuntime::Initialize(const conduit::Node &options)
{
    int rank = 0;
#if ASCENT_MPI_ENABLED
    if(!options.has_child("mpi_comm") ||
       !options["mpi_comm"].dtype().is_integer())
    {
        ASCENT_ERROR("Missing Ascent::open options missing MPI communicator (mpi_comm)");
    }

    flow::Workspace::set_default_mpi_comm(options["mpi_comm"].as_int());
    MPI_Comm comm = MPI_Comm_f2c(options["mpi_comm"].to_int());
    MPI_Comm_rank(comm,&rank);

#endif

    m_runtime_options = options;

    // standard flow filters
    flow::filters::register_builtin();
    // filters for ascent flow runtime.
    runtime::filters::register_builtin();

    if(options.has_path("web/stream") &&
       options["web/stream"].as_string() == "true" &&
       rank == 0)
    {

        if(options.has_path("web/document_root"))
        {
            m_web_interface.SetDocumentRoot(options["web/document_root"].as_string());
        }

        m_web_interface.Enable();
    }

    Node msg;
    this->Info(msg["info"]);
    ascent::about(msg["about"]);
    m_web_interface.PushMessage(msg);
}

//-----------------------------------------------------------------------------
void
FlowRuntime::Info(conduit::Node &out)
{
    out.set(m_info);
}

//-----------------------------------------------------------------------------
conduit::Node &
FlowRuntime::Info()
{
    return m_info;
}

//-----------------------------------------------------------------------------
void
FlowRuntime::Cleanup()
{

}

//-----------------------------------------------------------------------------
void
FlowRuntime::RegisterCallback(const std::string &callback_name,
                              void (*callback_function)(void))
{

}

//-----------------------------------------------------------------------------
void
FlowRuntime::RegisterCallback(const std::string &callback_name,
                              bool (*callback_function)(void))
{

}

//-----------------------------------------------------------------------------
void
FlowRuntime::Publish(const conduit::Node &data)
{
    // create our own tree, with all data zero copied.
    m_data.set_external(data);
}

//-----------------------------------------------------------------------------
void
FlowRuntime::ResetInfo()
{
    m_info.reset();
    m_info["runtime/type"] = "flow";
    m_info["runtime/options"] = m_runtime_options;
}

//-----------------------------------------------------------------------------
void
FlowRuntime::ConnectSource()
{
    // note: if the reg entry for data was already added
    // the set_external updates everything,
    // we don't need to remove and re-add.
    if(!w.registry().has_entry("_ascent_input_data"))
    {
        w.registry().add<Node>("_ascent_input_data",
                               &m_data);
    }

    if(!w.graph().has_filter("source"))
    {
       Node p;
       p["entry"] = "_ascent_input_data";
       w.graph().add_filter("registry_source","source",p);
    }
}

//-----------------------------------------------------------------------------
void
FlowRuntime::Execute(const conduit::Node &actions)
{
    ResetInfo();
    // make sure we always have our source data
    ConnectSource();
    // Loop over the actions
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        string action_name = action["action"].as_string();

        ASCENT_INFO("Executing " << action_name);

        // implement actions

        if(action_name == "add_filter")
        {
            if(action.has_child("params"))
            {
                w.graph().add_filter(action["type_name"].as_string(),
                                     action["name"].as_string(),
                                     action["params"]);
            }
            else
            {
                w.graph().add_filter(action["type_name"].as_string(),
                                     action["name"].as_string());
            }
        }
        else if( action_name == "add_filters")
        {
            w.graph().add_filters(action["filters"]);
        }
        else if( action_name == "connect")
        {
            if(action.has_child("port"))
            {
                w.graph().connect(action["src"].as_string(),
                                  action["dest"].as_string(),
                                  action["port"].as_string());
            }
            else
            {
                // if no port, assume input 0
                w.graph().connect(action["src"].as_string(),
                                  action["dest"].as_string(),
                                  0);
            }
        }
        else if( action_name == "add_connections")
        {
            w.graph().add_connections(action["connections"]);
        }
        else if( action_name == "add_graph")
        {
            w.graph().add_graph(action["graph"]);
        }
        else if( action_name == "load_graph")
        {
            w.graph().load(action["path"].as_string());
        }
        else if( action_name == "save_graph")
        {
            w.graph().save(action["path"].as_string());
        }
        else if( action_name == "execute")
        {
            w.info(m_info["flow_graph"]);
            w.execute();
            w.registry().reset();

            Node msg;
            this->Info(msg["info"]);
            ascent::about(msg["about"]);
            m_web_interface.PushMessage(msg);
        }
        else if( action_name == "reset")
        {
            w.reset();
        }

    }
}






//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



