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
/// file: alpine_flow_workspace.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_flow_workspace.hpp"

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
class Workspace::ExecutionPlan
{
    public:

        static void generate(Graph &g,
                             conduit::Node &traversals);
        
    private:
        ExecutionPlan();
        ~ExecutionPlan();
        
        static void bf_topo_sort_visit(Graph &graph,
                                       const std::string &filter_name,
                                       conduit::Node &tags,
                                       conduit::Node &tarv);
};

//-----------------------------------------------------------------------------
class Workspace::FilterFactory
{
public:

    static std::map<std::string,FilterFactoryMethod> &registered_types()
    {
        return m_filter_types;
    }
        
private:
    static std::map<std::string,FilterFactoryMethod> m_filter_types;
};

//-----------------------------------------------------------------------------
std::map<std::string,FilterFactoryMethod> Workspace::FilterFactory::m_filter_types;


//-----------------------------------------------------------------------------
Workspace::ExecutionPlan::ExecutionPlan()
{
    // empty
}

//-----------------------------------------------------------------------------
Workspace::ExecutionPlan::~ExecutionPlan()
{
    // empty
}


//-----------------------------------------------------------------------------
void
Workspace::ExecutionPlan::generate(Graph &graph,
                                   conduit::Node &traversals)
{   
    traversals.reset();

    Node snks;
    Node srcs;

    std::map<std::string,Filter*>::iterator itr;
    
    for(itr  = graph.m_filters.begin();
        itr != graph.m_filters.end();
        itr++)
    {
        Filter *f = itr->second;

        // check for snk
        if( !f->output_port() || 
             graph.edges_out(f->name()).number_of_children() == 0)
        {
            snks.append().set(f->name());
        }

        // check for src
        if( f->output_port() && 
            !graph.edges()["in"].has_child(f->name()) )
        {
            srcs.append().set(f->name());
        }
            
        
    }

    // init tags
    Node tags;
    for(itr  = graph.m_filters.begin();
        itr != graph.m_filters.end() ; 
        itr++)
    {
        Filter *f = itr->second;
        tags[f->name()].set_int32(0);

    }

    // execute bf traversal from each snk

    NodeConstIterator snk_itr(&snks);
    while(snk_itr.has_next())
    {
        std::string snk_name = snk_itr.next().as_string();

        Node snk_trav;
        bf_topo_sort_visit(graph, snk_name, tags, snk_trav);
        if(snk_trav.number_of_children() > 0)
        {
            traversals.append().set(snk_trav);
        }
    }
}


//-----------------------------------------------------------------------------
void
Workspace::ExecutionPlan::bf_topo_sort_visit(Graph &graph,
                                             const std::string &f_name,
                                             conduit::Node &tags,
                                             conduit::Node &trav)
{
    if( tags[f_name].as_int32() != 0 )
    {
        return;
    }

    int uref = 1;
    tags[f_name].set_int32(1);

    Filter *f = graph.m_filters[f_name];
    
    if(f->output_port())
    {
        int num_refs = graph.edges_out(f_name).number_of_children();
        uref = num_refs > 0 ? num_refs : 1;
    }
    
    if ( f->port_names().number_of_children() > 0 )
    {
        NodeConstIterator f_inputs(&graph.edges_in(f_name));
        
        while(f_inputs.has_next())
        {
            const Node &n_f_input = f_inputs.next();
            
            if(n_f_input.dtype().is_string())
            {
                std::string f_in_name = n_f_input.as_string();
                bf_topo_sort_visit(graph, f_in_name, tags, trav);
            }
            else // dangle?
            {
                uref = 0;
            }
        }
    }

    // conduit nodes keep insert order, so we can use
    // obj instead of list
    if(uref > 0)
    {
        trav[f_name] = uref;
    }

}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Workspace::Workspace()
:m_graph(this)
{

}

//-----------------------------------------------------------------------------
Workspace::~Workspace()
{

}

//-----------------------------------------------------------------------------
Graph &
Workspace::graph()
{
    return m_graph;
}

//-----------------------------------------------------------------------------
Registry &
Workspace::registry()
{
    return m_registry;
}



//-----------------------------------------------------------------------------
void
Workspace::execute()
{
    Node traversals;
    ExecutionPlan::generate(graph(),traversals);

    ALPINE_INFO(traversals.to_json());
    
    // execute traversals 
    NodeIterator travs_itr(&traversals);
    
    while(travs_itr.has_next())
    {
        NodeIterator trav_itr(&travs_itr.next());

        while(trav_itr.has_next())
        {
            Node &t = trav_itr.next();
            
            std::string  f_name = trav_itr.name();
            int          uref   = t.to_int32();
            Filter      *f      = graph().filters()[f_name];
        
            f->reset_inputs_and_output();

            // fetch inputs from reg, attach to filter's ports
            NodeConstIterator ports_itr = NodeConstIterator(&f->port_names());

            std::vector<std::string> f_i_names;

            while(ports_itr.has_next())
            {
                std::string port_name = ports_itr.next().as_string();
                std::string f_input_name = graph().edges_in(f_name)[port_name].as_string();
                f->set_input(port_name,&registry().fetch_data(f_input_name));
            }


            // execute 
            f->execute();

            // if has output, set output
            if(f->output_port())
            {
                
                registry().add_entry(f_name,
                                     f->output(),
                                     uref);
            }
            
            f->reset_inputs_and_output();
            
            // consume inputs
            ports_itr.to_front();
            while(ports_itr.has_next())
            {
                std::string port_name = ports_itr.next().as_string();
                std::string f_input_name = graph().edges_in(f_name)[port_name].as_string();
                registry().consume(f_input_name);
            }
        }
    }
    
    
}


//-----------------------------------------------------------------------------
void
Workspace::info(Node &out)
{
    out.reset();
    
    graph().info(out["graph"]);
    registry().info(out["registry"]);
}


//-----------------------------------------------------------------------------
std::string
Workspace::to_json()
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Workspace::print()
{
    ALPINE_INFO(to_json());
}

//-----------------------------------------------------------------------------
Filter *
Workspace::create_filter(const std::string &filter_type)
{
    if(!supports_filter_type(filter_type))
    {
        ALPINE_WARN("Cannot create unknown filter type: "
                    << filter_type);
        return NULL;
    }
    
    return FilterFactory::registered_types()[filter_type]();
}

//-----------------------------------------------------------------------------
bool
Workspace::supports_filter_type(const std::string &filter_type)
{
    std::map<std::string,FilterFactoryMethod>::const_iterator itr;
    itr = FilterFactory::registered_types().find(filter_type);
    return (itr != FilterFactory::registered_types().end());
}

//-----------------------------------------------------------------------------
void
Workspace::remove_filter_type(const std::string &filter_type)
{
    std::map<std::string,FilterFactoryMethod>::const_iterator itr;
    itr = FilterFactory::registered_types().find(filter_type);
    if(itr != FilterFactory::registered_types().end())
    {
        FilterFactory::registered_types().erase(filter_type);
    }
}
//-----------------------------------------------------------------------------
void
Workspace::register_filter_type(FilterFactoryMethod fr)
{
    
    // TODO: we want the interface info to be static ...
    
    // check that filter is valid by creating
    // an instance
    
    Filter *f = fr();
    
    // verify f provides proper interface declares
    
    Node i_test;
    Node v_info;
    
    std::string f_type_name = "(type_name missing!)";
    
    f->declare_interface(i_test);
    if(!Filter::verify_interface(i_test,v_info))
    {
        // if  the type name was provided, that helps improve 
        // the error message, so try to include it
        if(i_test.has_child("type_name") && 
           i_test["type_name"].dtype().is_string())
        {
            f_type_name = i_test["type_name"].as_string();
        }
        
        // failed interface verify ... 
        ALPINE_ERROR("filter type interface verify failed." << std::endl
                      << f_type_name   << std::endl
                      << "Details:" << std::endl
                      << v_info.to_json());
    }
        
    f_type_name =i_test["type_name"].as_string();
    
    // we no longer need this instance ...
    delete f;
    
    if(supports_filter_type(f_type_name))
    {
        ALPINE_ERROR("filter type named:"
                     << f_type_name 
                    << " is already registered");
    }
    
    FilterFactory::registered_types()[f_type_name] = fr;
}


//-----------------------------------------------------------------------------
void
Workspace::clear_supported_filter_types()
{
    FilterFactory::registered_types().clear();
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



