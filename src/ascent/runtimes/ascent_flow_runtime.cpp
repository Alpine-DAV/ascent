//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://software.llnl.gov/ascent/.
// 
// Please also read ascent/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
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
void
FlowRuntime::Cleanup()
{
}

//-----------------------------------------------------------------------------
void
FlowRuntime::Publish(const conduit::Node &data)
{
    // create our own tree, with all data zero copied.
    m_data.set_external(data);
    
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
FlowRuntime::ResetInfo()
{
    m_info.reset();
    m_info["runtime/type"] = "flow";
    m_info["runtime/options"] = m_runtime_options;
}


//-----------------------------------------------------------------------------
void
FlowRuntime::Execute(const conduit::Node &actions)
{
    ResetInfo();
    
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



