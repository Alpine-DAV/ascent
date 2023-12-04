//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_filter.cpp
///
//-----------------------------------------------------------------------------

#include "flow_filter.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

//-----------------------------------------------------------------------------
// flow includes
//-----------------------------------------------------------------------------
#include <flow_data.hpp>
#include <flow_registry.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>


using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
Filter::Filter()
: m_graph(NULL),
  m_out(NULL)
{
    // provide NULL as default data when output is not set
    // null pointer won't be reaped, concrete type doesn't matter,
    // except that we can't use void
    m_out = new DataWrapper<int>(NULL);
}

//-----------------------------------------------------------------------------
Filter::~Filter()
{
    if (m_out != NULL)
    {
        delete m_out;
    }
}


//-----------------------------------------------------------------------------
Graph &
Filter::graph()
{
    return *m_graph;
}

//-----------------------------------------------------------------------------
void
Filter::connect_input_port(const std::string &port_name,
                           Filter *filter)
{
    graph().connect(filter->name(),
                    this->name(),
                    port_name);
}

//-----------------------------------------------------------------------------
void
Filter::connect_input_port(int idx,
                           Filter *filter)
{
    graph().connect(filter->name(),
                    this->name(),
                    this->port_index_to_name(idx));
}


//-----------------------------------------------------------------------------
void
Filter::init(Graph *g,
             const std::string &name,
             const Node &p)
{
    m_graph = g;
    m_props["name"] = name;
    declare_interface(interface());

    Node &n_iface = properties()["interface"];

    // we need to fill in any missing props that
    // interface may const fetch

    if( !n_iface.has_child("default_params") )
    {
        n_iface["default_params"] = DataType::empty();
    }

    if( !n_iface.has_child("port_names") )
    {
        n_iface["port_names"] = DataType::empty();
    }


    params().update(default_params());
    params().update(p);
}

//-----------------------------------------------------------------------------
std::string
Filter::name() const
{
    return m_props["name"].as_string();
}

//-----------------------------------------------------------------------------
std::string
Filter::type_name() const
{
    return properties()["interface/type_name"].as_string();
}

//-----------------------------------------------------------------------------
const Node &
Filter::default_params() const
{
    return properties()["interface/default_params"];
}

//-----------------------------------------------------------------------------
const Node &
Filter::port_names() const
{
    return properties()["interface/port_names"];
}

//-----------------------------------------------------------------------------
bool
Filter::output_port() const
{
    return properties()["interface/output_port"].as_string() == "true";
}

//-----------------------------------------------------------------------------
bool
Filter::has_port(const std::string &port_name) const
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
Node &
Filter::properties()
{
    return m_props;
}

//-----------------------------------------------------------------------------
const Node &
Filter::properties() const
{
    return m_props;
}


//-----------------------------------------------------------------------------
const Node &
Filter::interface() const
{
    return m_props["interface"];
}

//-----------------------------------------------------------------------------
// note this one is private.
//-----------------------------------------------------------------------------
Node &
Filter::interface()
{
    return m_props["interface"];
}


//-----------------------------------------------------------------------------
Node &
Filter::params()
{
    return m_props["params"];
}


//-----------------------------------------------------------------------------
bool
Filter::verify_params(const Node &, // unused: params,
                      Node &info)
{
    info.reset();
    return true;
}


//-----------------------------------------------------------------------------
bool
Filter::verify_interface(const Node &i,
                         Node &info)
{
    bool res = true;
    info.reset();


    if(!i.has_child("type_name") || !i["type_name"].dtype().is_string())
    {
        std::string msg = "interface missing 'type_name' = {string}";
        info["errors"].append().set(msg);
        res = false;
    }

    if(!i.has_child("output_port") ||
       !i["output_port"].dtype().is_string() )
    {
        std::string msg = "interface missing 'output_port' = {\"true\" | \"false\"}";
        info["errors"].append().set(msg);
        res = false;
    }
    else
    {
        std::string oport = i["output_port"].as_string();

        if(oport != "true" && oport !="false")
        {

            std::string msg = "interface 'output_port' is \"" + oport + "\""
                              ", expected {\"true\" | \"false\"}";
            info["errors"].append().set(msg);
            res = false;
        }
    }

    if(i.has_child("port_names"))
    {
        NodeConstIterator itr(&i["port_names"]);
        int idx = 0;
        while(itr.has_next())
        {
            const Node &curr = itr.next();
            if(!curr.dtype().is_string())
            {
                ostringstream oss;
                oss <<  "interface port_name at index " << idx << " is not a string";
                info["errors"].append().set(oss.str());
                res = false;
            }
            idx++;
        }
    }


    if(i.has_child("default_params"))
    {
        info["info"].append().set("interface provides 'default_params'");
    }


    return res;
}


//-----------------------------------------------------------------------------
Data &
Filter::input(const std::string &port_name)
{
    if(!has_port(port_name) )
    {
        CONDUIT_ERROR( detailed_name()
                      << "does not have an input port named: " << port_name);
    }

    return *m_inputs[port_name];
}

//-----------------------------------------------------------------------------
Data &
Filter::input(int idx)
{
    return *m_inputs[port_index_to_name(idx)];
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// private helpers
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Data *
Filter::fetch_input(const std::string &port_name)
{
    if(!has_port(port_name) )
    {
        CONDUIT_ERROR( detailed_name()
                      << "does not have an input port named: " << port_name);
    }

    return m_inputs[port_name];
}

//-----------------------------------------------------------------------------
Data *
Filter::fetch_input(int port_idx)
{
    return m_inputs[port_index_to_name(port_idx)];
}

//-----------------------------------------------------------------------------
void
Filter::set_output(Data &data)
{
    if(m_out != NULL)
    {
        delete m_out;
        m_out = NULL;
    }

    m_out = data.wrap(data.data_ptr());
}


//-----------------------------------------------------------------------------
Data &
Filter::output()
{
    return *m_out;
}



//-----------------------------------------------------------------------------
void
Filter::set_input(const std::string &port_name,
                  Data *data)
{
    m_inputs[port_name] = data;
}


//-----------------------------------------------------------------------------
std::string
Filter::detailed_name() const
{
    return name() + "(type: " + type_name() + ")";

}

//-----------------------------------------------------------------------------
int
Filter::number_of_input_ports() const
{
    return port_names().number_of_children();
}

//-----------------------------------------------------------------------------
std::string
Filter::port_index_to_name(int idx) const
{
    const Node &pnames = port_names();

    index_t nports = number_of_input_ports();

    if(idx > nports || idx < 0)
    {
        CONDUIT_ERROR("Fasiled to find input port name for index: " << idx
                     << " " << detailed_name() << " has " << nports
                     << " input port ports ");
    }

    return pnames[idx].as_string();
}


//-----------------------------------------------------------------------------
void
Filter::reset_inputs_and_output()
{
    // inputs aren't owned
    m_inputs.clear();

    // output pointer to container is owned
    if(m_out != NULL)
    {
        delete m_out;
        m_out = NULL;
    }
}


//-----------------------------------------------------------------------------
void
Filter::info(Node &out) const
{
    out.reset();
    out.set(m_props);

    Node &f_inputs = out["inputs"];

    std::map<std::string,Data*>::const_iterator itr;
    for(itr = m_inputs.begin(); itr != m_inputs.end(); itr++)
    {
        itr->second->info(f_inputs[itr->first]);
    }

    if(m_out != NULL)
    {
        m_out->info(out["output"]);
    }
    else
    {
        out["output"] = DataType::empty();
    }
}


//-----------------------------------------------------------------------------
std::string
Filter::to_json() const
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
std::string
Filter::to_yaml() const
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_yaml_stream(oss);
    return oss.str();
}


//-----------------------------------------------------------------------------
void
Filter::print() const
{
    CONDUIT_INFO(to_yaml());
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------



