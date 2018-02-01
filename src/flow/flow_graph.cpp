//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Ascent. 
// 
// For details, see: http://ascent.readthedocs.io/.
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
/// file: flow_graph.cpp
///
//-----------------------------------------------------------------------------

#include "flow_graph.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit_relay.hpp>


//-----------------------------------------------------------------------------
// flow includes
//-----------------------------------------------------------------------------
#include <flow_workspace.hpp>


using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Graph::Graph(Workspace *w)
:m_workspace(w),
 m_filter_count(0)
{
    init();
}

//-----------------------------------------------------------------------------
Graph::~Graph()
{
    reset();
}


//-----------------------------------------------------------------------------
Workspace &
Graph::workspace()
{
    return *m_workspace;
}

//-----------------------------------------------------------------------------
void 
Graph::reset()
{
    // delete all filters

    std::map<std::string,Filter*>::iterator itr;
    for(itr = m_filters.begin(); itr != m_filters.end(); itr++)
    {
        delete itr->second;
    }

    m_filters.clear();
    m_edges.reset();
    init();

}

//-----------------------------------------------------------------------------
void 
Graph::init()
{
    // init edges data 
    m_edges["in"];
    m_edges["out"];

}


//-----------------------------------------------------------------------------
Filter *
Graph::add_filter(const std::string &filter_type,
                  const std::string &filter_name)
{
    Node filter_params;
    return add_filter(filter_type, filter_name, filter_params);
}


//-----------------------------------------------------------------------------
Filter *
Graph::add_filter(const std::string &filter_type,
                  const std::string &filter_name,
                  const Node &filter_params)
{
    if(has_filter(filter_name))
    {
        CONDUIT_WARN("Cannot create filter, filter named: " << filter_name
                     << " already exists in Graph");
        return NULL;
    }
    
    Filter *f = Workspace::create_filter(filter_type);
    
    f->init(this,
            filter_name,
            filter_params);
    
    Node v_info;
    if(!f->verify_params(filter_params,v_info))
    {
        std::string f_name = f->detailed_name();
        // cleanup f ... 
        delete f;
        CONDUIT_WARN("Cannot create filter " << f_name 
                    << " because verify_params failed." << std::endl
                    << "Details:" << std::endl
                    << v_info.to_json());
        return NULL;
    }
    
    
    m_filters[filter_name] = f;
    
    NodeConstIterator ports_itr = f->port_names().children();

    while(ports_itr.has_next())
    {
        std::string port_name = ports_itr.next().as_string();
        m_edges["in"][filter_name][port_name] = DataType::empty();
    }
    
    if(f->output_port())
    {
        m_edges["out"][filter_name] = DataType::list();
    }
    
    m_filter_count++;
    
    return f;
}

//-----------------------------------------------------------------------------
Filter *
Graph::add_filter(const std::string &filter_type)
{
    ostringstream oss;
    oss << "f_" << m_filter_count;
    Node filter_params;
    return add_filter(filter_type, oss.str(), filter_params);
}

//-----------------------------------------------------------------------------
Filter *
Graph::add_filter(const std::string &filter_type,
                  const Node &filter_params)
{
    ostringstream oss;
    oss << "f_" << m_filter_count;
    return add_filter(filter_type, oss.str(), filter_params);
}



//-----------------------------------------------------------------------------
void 
Graph::connect(const std::string &src_name,
               const std::string &des_name,
               const std::string &port_name)
{
    // make sure we have a filter with the given name
    
    if(!has_filter(src_name))
    {
        CONDUIT_WARN("source filter named: " << src_name
                    << " does not exist in Filter Graph");
        return;
    }

    if(!has_filter(des_name))
    {
        CONDUIT_WARN("destination filter named: " << des_name
                    << " does not exist in Filter Graph");
        return;
    }


    Filter *des_filter = m_filters[des_name];

    // make sure it has an input port with the given name
    if(!des_filter->has_port(port_name))
    {
        CONDUIT_WARN("destination filter: "
                     << des_filter->detailed_name()
                     << " does not have input port named:"
                     << port_name);
        return;
    }
    
    m_edges["in"][des_name][port_name] = src_name;
    m_edges["out"][src_name].append().set(des_name);
}

//-----------------------------------------------------------------------------
void 
Graph::connect(const std::string &src_name,
               const std::string &des_name,
               int port_idx)
{
    if(!has_filter(des_name))
    {
        CONDUIT_WARN("destination filter named: " << des_name
                    << " does not exist in Filter Graph ");
        return;
    }

    Filter *des_filter = m_filters[des_name];
    std::string port_name = des_filter->port_index_to_name(port_idx);


    connect(src_name,des_name,port_name);
}




//-----------------------------------------------------------------------------
bool
Graph::has_filter(const std::string &name)
{
    std::map<std::string,Filter*>::iterator itr = m_filters.find(name);
    return itr != m_filters.end();
}

//-----------------------------------------------------------------------------
void
Graph::remove_filter(const std::string &name)
{
    if(!has_filter(name))
    {
        CONDUIT_WARN("filter named: " << name
                     << " does not exist in Filter Graph");
        return;
    }

    // remove from m_filters, and prune edges
    std::map<std::string,Filter*>::iterator itr = m_filters.find(name);
    
    delete itr->second;
    
    m_filters.erase(itr);
    
    m_edges["in"].remove(name);
    m_edges["out"].remove(name);
}

//-----------------------------------------------------------------------------
const Node &
Graph::edges() const
{
    return m_edges;
}

//-----------------------------------------------------------------------------
const Node &
Graph::edges_out(const std::string &f_name) const
{
    return edges()["out"][f_name];
}

//-----------------------------------------------------------------------------
const Node &
Graph::edges_in(const std::string &f_name) const
{
    return edges()["in"][f_name];
}


//-----------------------------------------------------------------------------
std::map<std::string,Filter*>  &
Graph::filters()
{
    return m_filters;
}


//-----------------------------------------------------------------------------
void
Graph::filters(Node &out) const
{
    out.reset();
    std::map<std::string,Filter*>::const_iterator itr;
    for(itr = m_filters.begin(); itr != m_filters.end(); itr++)
    {
        Filter *f_ptr = itr->second;
        Node &f_info = out[itr->first];
        f_info["type_name"] = f_ptr->type_name();
        
        if(f_ptr->params().number_of_children() > 0)
        {
            f_info["params"] = f_ptr->params();
        }
    }
}


//-----------------------------------------------------------------------------
void
Graph::connections(Node &out) const
{
    out.reset();
    NodeConstIterator edges_itr = edges()["in"].children();
    while(edges_itr.has_next())
    {
        const Node &edge = edges_itr.next();
        std::string dest_name = edges_itr.name();
        NodeConstIterator ports_itr = edge.children();
        while(ports_itr.has_next())
        {
            const Node &port = ports_itr.next();
            if(port.dtype().is_string())
            {
                std::string port_name = ports_itr.name();
                std::string src_name  = port.as_string();
                Node &edge = out.append();
                edge["src"]  = src_name;
                edge["dest"] = dest_name;
                edge["port"] = port_name;
            }
        }
    }
}



//-----------------------------------------------------------------------------
void
Graph::add_filters(const Node &filters)
{
    CONDUIT_INFO(filters.to_json());

    NodeConstIterator filters_itr = filters.children();

    // first make sure we have only supported filters.
    bool ok = true;
    ostringstream oss;
    
    while(filters_itr.has_next())
    {
        const Node &f_info = filters_itr.next();
        std::string f_name = filters_itr.name();
        
        if(!f_info.has_child("type_name") ||
           !f_info["type_name"].dtype().is_string())
        {
            oss << "Filter '" 
                << f_name 
                << "' is missing required 'type_name' entry"
                << std::endl;
            ok = false;
        }
        else 
        {
            std::string f_type = f_info["type_name"].as_string();
            if(!Workspace::supports_filter_type(f_type))
            {
            
                oss << "Workspace does not support filter type "
                    << "'" << f_type << "' "
                    << "(filter name: '" << f_name << "')"
                    << std::endl;
                ok = false;
            }
        }
    }
    
    // provide one error message with all issues discovered
    if(!ok)
    {
        CONDUIT_ERROR(oss.str());
        return;
    }
    
    filters_itr.to_front();

    while(filters_itr.has_next())
    {
        const Node &f_info = filters_itr.next();
        std::string f_name = filters_itr.name();
        std::string f_type = f_info["type_name"].as_string();

        if(f_info.has_child("params"))
        {
            add_filter(f_type,f_name,f_info["params"]);
        }
        else
        {
            add_filter(f_type,f_name);
        }
    }    
}

//-----------------------------------------------------------------------------
void
Graph::add_connections(const Node &conns)
{
    CONDUIT_INFO(conns.to_json());

    NodeConstIterator conns_itr = conns.children();
    while(conns_itr.has_next())
    {
        const Node &edge = conns_itr.next();

        bool ok = true;
        ostringstream oss;
        if(!edge.has_child("src") ||
           !edge["src"].dtype().is_string())
        {
            oss << "Connection is missing required 'src' entry" << std::endl;
            ok = false;
        }

        if(!edge.has_child("dest") ||
           !edge["dest"].dtype().is_string())
        {
            oss << "Connection is missing required 'dest' entry" << std::endl;
            ok = false;
        }
        
        if(!ok)
        {
            CONDUIT_ERROR(oss.str());
            return;
        }
  
        
        if(edge.has_child("port"))
        {
            connect(edge["src"].as_string(),
                    edge["dest"].as_string(),
                    edge["port"].as_string());
        }
        else
        {
            connect(edge["src"].as_string(),
                    edge["dest"].as_string(),
                    0);
        }
    }
}


//-----------------------------------------------------------------------------
void
Graph::add_graph(const Graph &g)
{
    Node n;
    g.info(n);
    add_graph(n);
}

//-----------------------------------------------------------------------------
void
Graph::add_graph(const Node &g)
{
    if(g.has_child("filters"))
    {
        add_filters(g["filters"]);
    }

    if(g.has_child("connections"))
    {
        add_connections(g["connections"]);
    }
}



//-----------------------------------------------------------------------------
void
Graph::save(const std::string &path,const std::string &protocol)
{
    Node out;
    save(out);
    conduit::relay::io::save(out,path,protocol);
}

//-----------------------------------------------------------------------------
void
Graph::save(Node &out)
{
    out.reset();
    info(out);
}


//-----------------------------------------------------------------------------
void
Graph::load(const std::string &path, const std::string &protocol)
{
    Node n;
    conduit::relay::io::load(path,protocol,n);
    load(n);
}


//-----------------------------------------------------------------------------
void
Graph::load(const Node &n)
{
    reset();
    add_graph(n);
}




//-----------------------------------------------------------------------------
void
Graph::info(Node &out) const
{
    out.reset();
    filters(out["filters"]);
    connections(out["connections"]);
}


//-----------------------------------------------------------------------------
std::string
Graph::to_json() const
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}


//-----------------------------------------------------------------------------
std::string
Graph::to_dot() const
{
    Node out;
    info(out);

    ostringstream oss;
    
    // traverse conns to create a dot graph;
    oss << "digraph {" << std::endl;
    
    
    NodeConstIterator itr = out["filters"].children();
    while(itr.has_next())
    {
        const Node &f= itr.next();
        std::string f_name = itr.name();
        oss << "  "
            << f_name 
            << " [label=\"" << f_name 
            << "(" << f["type_name"].as_string() << ")" 
            << "\"];" << std::endl;
    }
    
    itr = out["connections"].children();
    
    while(itr.has_next())
    {
        const Node &c= itr.next();
        oss << "  "
            << c["src"].as_string() 
            << " -> " 
            << c["dest"].as_string()
            << "[ label=\"" << c["port"].as_string() << "\" ]"
            << ";"
            << std::endl;
    }
    
    oss << "}" << std::endl;
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Graph::print() const
{
    CONDUIT_INFO(to_json());
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------




