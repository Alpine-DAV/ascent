//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Alpine. 
// 
// For details, see: http://software.llnl.gov/alpine/.
// 
// Please also read alpine/LICENSE
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
/// file: alpine_flow_filter.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_flow_filter.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>


//-----------------------------------------------------------------------------
// alpine includes
//-----------------------------------------------------------------------------
#include <alpine_logging.hpp>
#include <alpine_flow_data.hpp>
#include <alpine_flow_registry.hpp>
#include <alpine_flow_graph.hpp>
#include <alpine_flow_workspace.hpp>


using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

//-----------------------------------------------------------------------------
// -- begin alpine::flow --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
Filter::Filter()
: m_graph(NULL)
{

}


//-----------------------------------------------------------------------------
Filter::~Filter()
{
    
}


//-----------------------------------------------------------------------------
Graph &
Filter::graph()
{
    return *m_graph;
}

//-----------------------------------------------------------------------------
void
Filter::init(Graph *g,
             const std::string &name,
             const Node &p)
{
    m_graph = g;
    m_props["name"] = name;
    params().update(default_params());
    params().update(p);
}

//-----------------------------------------------------------------------------
std::string
Filter::name()
{
    return m_props["name"].as_string();
}

//-----------------------------------------------------------------------------
std::string
Filter::type_name()
{
    return m_props["type_name"].as_string();
}

//-----------------------------------------------------------------------------
const Node &
Filter::default_params()
{
    return m_props["default_params"];
}

//-----------------------------------------------------------------------------
const Node &
Filter::port_names()
{
    return m_props["port_names"];
}

//-----------------------------------------------------------------------------
bool
Filter::output_port()
{
    return m_props["output_port"].as_string() == "true";
}

//-----------------------------------------------------------------------------
Node &
Filter::properties()
{
    return m_props;
}


//-----------------------------------------------------------------------------
Node &
Filter::params()
{
    return m_props["params"];
}

//-----------------------------------------------------------------------------
Data &
Filter::input(const std::string &port_name)
{
    if(!has_port(port_name) )
    {
        ALPINE_ERROR( name() << "(type: " << type_name() << ") "
                      << "does not have an input port named: " << port_name);
    }

    return m_inputs[port_name];
}


//-----------------------------------------------------------------------------
void
Filter::set_input(const std::string &port_name, 
                  Data ds)
{
    m_inputs[port_name] = ds;
}



//-----------------------------------------------------------------------------
bool
Filter::has_port(const std::string &port_name)
{
    bool found = false;

    NodeConstIterator itr(&port_names());
    while(itr.has_next() && ! found)
    {
        found = (port_name == itr.next().as_string());
    }

    return found;
}

//-----------------------------------------------------------------------------
Data &
Filter::output()
{
    return m_out;
}

//-----------------------------------------------------------------------------
void
Filter::reset_inputs_and_output()
{
    m_inputs.clear();
    m_out = Data();
}


//-----------------------------------------------------------------------------
void
Filter::info(Node &out)
{
    out.reset();
    out.set(m_props);
    
    Node &f_inputs = out["inputs"];
    
    std::map<std::string,Data>::iterator itr;
    for(itr = m_inputs.begin(); itr != m_inputs.end(); itr++)
    {   
        itr->second.info(f_inputs[itr->first]);
    }
    
    m_out.info(out["output"]);
}


//-----------------------------------------------------------------------------
std::string
Filter::to_json()
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Filter::print()
{
    ALPINE_INFO(to_json());
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine::flow --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



