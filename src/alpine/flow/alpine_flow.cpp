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
/// file: alpine_flow.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_flow.hpp"

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
//-----------------------------------------------------------------------------
//
// Dataset::Value
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class Dataset::Value
{
public:
     Value();
     Value(Value &v);
     Value &operator=(Value &v);
    ~Value();
    

    Node *m_node_ptr;
 
};

//-----------------------------------------------------------------------------
Dataset::Value::Value()
:m_node_ptr(NULL)
{

}

//-----------------------------------------------------------------------------
Dataset::Value::Value(Dataset::Value &v)
:m_node_ptr(v.m_node_ptr)
{

}


//-----------------------------------------------------------------------------
Dataset::Value &
Dataset::Value::operator=(Dataset::Value &v)
{
    if(&v != this)
    {
        m_node_ptr = v.m_node_ptr;
    }
    return *this;
}


//-----------------------------------------------------------------------------
Dataset::Value::~Value()
{

}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Dataset
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Dataset::Dataset()
:m_value(NULL)
{
    m_value = new Value();
}

//-----------------------------------------------------------------------------
Dataset::Dataset(Dataset &ds)
:m_value(NULL)
{
    m_value = new Value(*ds.m_value);
}

//-----------------------------------------------------------------------------
Dataset &
Dataset::operator=(Dataset &v)
{
    if(&v != this)
    {
        m_value->m_node_ptr = v.m_value->m_node_ptr;
    }
    return *this;
}


//-----------------------------------------------------------------------------
Dataset::~Dataset()
{
    delete m_value;
}


//-----------------------------------------------------------------------------
void
Dataset::set(Node *ds)
{
    m_value->m_node_ptr = ds;
}

//-----------------------------------------------------------------------------
void
Dataset::release()
{
    if(m_value->m_node_ptr != NULL)
    {
        delete m_value->m_node_ptr;
        m_value->m_node_ptr = NULL;
    }
    
}

//-----------------------------------------------------------------------------
conduit::Node *
Dataset::as_node_ptr()
{
    return m_value->m_node_ptr;
}

//-----------------------------------------------------------------------------
Dataset::operator Node *()
{
    return as_node_ptr();
}

//-----------------------------------------------------------------------------
void
Dataset::info(Node &out)
{
    out.reset();
    out["node_ptr"] = (index_t) m_value->m_node_ptr;
}



//-----------------------------------------------------------------------------
std::string
Dataset::to_json()
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Dataset::print()
{
    ALPINE_INFO(to_json());
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Registry::Map
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class Registry::Map
{
public:
    
    class Entry
    {
        public:
                 Entry();
                 ~Entry();

                 Dataset &dataset();

                 void     set_ref_count(int ref_count);
                 int      ref_count();
                 int      dec_ref_count();
            
                
        private:
            Dataset  m_dataset;
            int      m_ref_count;
        
    };
 
    Map();
   ~Map();

    Entry &fetch(const std::string &key);
    int    dec_ref_count(const std::string &key);
    void   info(Node &out);
    
private:
    std::map<std::string,Entry> m_entries;
    
};

//-----------------------------------------------------------------------------

Registry::Map::Entry::Entry()
:m_dataset(),
 m_ref_count(-1)
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Entry::~Entry()
{
    // empty
}

//-----------------------------------------------------------------------------
Dataset &
Registry::Map::Entry::dataset()
{
    return m_dataset;
}


//-----------------------------------------------------------------------------
void
Registry::Map::Entry::set_ref_count(int ref_count)
{
    m_ref_count = ref_count;
}

//-----------------------------------------------------------------------------
int
Registry::Map::Entry::ref_count()
{
    return m_ref_count;
}

//-----------------------------------------------------------------------------
int
Registry::Map::Entry::dec_ref_count()
{
    if(m_ref_count > 0)
    {
        m_ref_count--;
    }
    
    if(m_ref_count == 0)
    {
        m_dataset.release();
    }

    return m_ref_count;
}

//-----------------------------------------------------------------------------

Registry::Map::Map()
{
    
}

//-----------------------------------------------------------------------------
Registry::Map::~Map()
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Entry &
Registry::Map::fetch(const std::string &key)
{
    // todo, runtime error if does not exist?
    return m_entries[key];
}


//-----------------------------------------------------------------------------
int
Registry::Map::dec_ref_count(const std::string &key)
{
    int res = m_entries[key].dec_ref_count();
    if(res == 0)
    {
        m_entries.erase(key);
    }
    return res;
}


//-----------------------------------------------------------------------------
void
Registry::Map::info(Node &out)
{
    out.reset();

    std::map<std::string,Entry>::iterator itr;
    
    for(itr = m_entries.begin(); itr != m_entries.end(); itr++)
    {
        Entry &ent = itr->second;
        out[itr->first]["ref_count"] = ent.ref_count();
        ent.dataset().info(out[itr->first]["dataset"]);
    }
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
Registry::Registry()
:m_map(NULL)
{
    m_map = new Map();

}

//-----------------------------------------------------------------------------
Registry::~Registry()
{
    delete m_map;
}


//-----------------------------------------------------------------------------
void
Registry::add_entry(const std::string &key, 
                    conduit::Node *node,
                    int ref_count)
{
    // TODO: need seat belts here if key already exists?
    
    
    Registry::Map::Entry &ent = m_map->fetch(key);
    ent.dataset().set(node);
    ent.set_ref_count(ref_count);
    
    // debugging
    print();
}

//-----------------------------------------------------------------------------
Dataset &
Registry::fetch_entry(const std::string &key)
{
    // should map throw runtime error if key doesn't exist?
    return m_map->fetch(key).dataset();;
}

//-----------------------------------------------------------------------------
void
Registry::dec_entry_ref_count(const std::string &key)
{
    Registry::Map::Entry &ent = m_map->fetch(key);

    ent.dec_ref_count();
    // debugging
    print();
}



//-----------------------------------------------------------------------------
void
Registry::info(Node &out)
{
    m_map->info(out);
}

//-----------------------------------------------------------------------------
std::string
Registry::to_json()
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Registry::print()
{
    ALPINE_INFO(to_json());
}



//-----------------------------------------------------------------------------
Filter::Filter()
{

}


//-----------------------------------------------------------------------------
Filter::~Filter()
{
    
}


//-----------------------------------------------------------------------------
void
Filter::init()
{
    m_props["params"] = m_props["default_params"];
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
Filter::input_port_names()
{
    return m_props["input_port_names"];
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
Dataset &
Filter::input(const std::string &port_name)
{
    // TODO error checking ...
    return m_inputs[port_name];
}


//-----------------------------------------------------------------------------
void
Filter::set_input(const std::string &port_name, 
                  Dataset &ds)
{
    m_inputs[port_name] = ds;
}

//-----------------------------------------------------------------------------
bool
Filter::has_input_port(const std::string &port_name)
{
    bool found = false;

    NodeConstIterator itr(&input_port_names());
    while(itr.has_next() && ! found)
    {
        found = (port_name == itr.next().as_string());
    }

    return found;
}

//-----------------------------------------------------------------------------
Dataset &
Filter::output()
{
    return m_out;
}

//-----------------------------------------------------------------------------
void
Filter::info(Node &out)
{
    out.reset();
    out.set(m_props);
    
    Node &f_inputs = out["inputs"];
    
    std::map<std::string,Dataset>::iterator itr;
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
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
FilterGraph::FilterGraph()
{
    
}

//-----------------------------------------------------------------------------
FilterGraph::~FilterGraph()
{
    
}


//-----------------------------------------------------------------------------
void
FilterGraph::add_filter(const std::string &name,
                        Filter *f)
{
    if(has_filter(name))
    {
        ALPINE_ERROR("filter named: " << name
                       << "already exists in FilterGraph");
        return;
    }
    
    m_filters[name] = f;
    
    NodeConstIterator itr(&f->input_port_names());

    while(itr.has_next())
    {
        std::string port_name = itr.next().as_string();
        m_edges["in"][name][port_name] = DataType::empty();
    }
    
    if(f->output_port())
    {
        m_edges["out"][name] = DataType::list();
    }
}

//-----------------------------------------------------------------------------
void 
FilterGraph::connect(const std::string &src_name,
                     const std::string &des_name,
                     const std::string &port_name)
{
    // make sure we have a filter with the given name
    
    if(!has_filter(src_name))
    {
        ALPINE_ERROR("source filter named: " << src_name
                       << "does not exist in FilterGraph");
        return;
    }

    if(!has_filter(des_name))
    {
        ALPINE_ERROR("destination filter named: " << des_name
                       << "does not exist in FilterGraph");
        return;
    }


    Filter *src_filter = m_filters[src_name];
    Filter *des_filter = m_filters[des_name];

    // make sure it has an input port with the given name
    if(!des_filter->has_input_port(port_name))
    {
        ALPINE_ERROR("destination filter named: " << des_name
                       << "does not have input port named:"
                       << port_name);
        return;
    }
    
    m_edges["in"][des_name][port_name] = src_name;
    m_edges["out"][src_name].append().set(des_name);
}


//-----------------------------------------------------------------------------
bool
FilterGraph::has_filter(const std::string &name)
{
    std::map<std::string,Filter*>::iterator itr = m_filters.find(name);
    return itr != m_filters.end();
}

//-----------------------------------------------------------------------------
void
FilterGraph::remove_filter(const std::string &name)
{
    if(!has_filter(name))
    {
        ALPINE_ERROR("filter named: " << name
                       << "does not exist in FilterGraph");
        return;
    }

    // remove from m_filters, and prune edges
    std::map<std::string,Filter*>::iterator itr = m_filters.find(name);
    m_filters.erase(itr);
    
    m_edges["in"].remove(name);
    m_edges["out"].remove(name);
}

//-----------------------------------------------------------------------------
void
FilterGraph::info(Node &out)
{
    out.reset();
    Node &filts = out["filters"];
    
    std::map<std::string,Filter*>::iterator itr;
    for(itr = m_filters.begin(); itr != m_filters.end(); itr++)
    {
        itr->second->info(filts[itr->first]);
    }

    out["edges"] = m_edges;

}


//-----------------------------------------------------------------------------
std::string
FilterGraph::to_json()
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
FilterGraph::print()
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



