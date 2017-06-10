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
/// file: alpine_flow_registry.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_flow_registry.hpp"

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
// Registry::Map
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

class Registry::Map
{
public:
    // key    to refs_needed + Data
    // void*  to  data
    // data has its own refs_needed

    class Ref
    {
        public:
             Ref(int refs_needed=-1);
            ~Ref();

             bool tracked() const;
             int  pending() const;
             int  dec();
             int  inc(int amt=0);
             
             void set_pending(int pending);

        private:
            int  m_pending;
    };


    class Value
    {
        public:

            Value(Data &data, int refs_needed);
            ~Value();

            Data *data();
            Ref           *ref();
 
            void          *data_ptr();
 
        private:
            Ref            m_ref;
            Data *m_data;
    };

    class Entry
    {
        public:
                 Entry(Value *, int refs_needed);
                 ~Entry();

                 Value          *value();
                 Data  *data();
                 Ref            *ref();

        private:
            Ref    m_ref;
            Value *m_value;

        
    };

 
    Map();
   ~Map();

    void   add(const std::string &key,
               Data &d,
               int refs_needed=-1);

    bool   has_entry(const std::string &key);
    bool   has_value(void *data_ptr);
        
    Entry *fetch_entry(const std::string &key);
    
    Value *fetch_value(void *data_ptr);
    
    void   dec(const std::string &key);
    
    void   info(Node &out);
    
    void   reset();
    
private:

    std::map<void*,Value*>         m_values;
    std::map<std::string,Entry*>   m_entries;

};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Registry::Map::Ref Class
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Registry::Map::Ref::Ref(int refs_needed)
:m_pending(refs_needed)
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Ref::~Ref()
{
    // empty
}

//-----------------------------------------------------------------------------
int
Registry::Map::Ref::pending() const
{
    return m_pending;
}

//-----------------------------------------------------------------------------
void
Registry::Map::Ref::set_pending(int pending)
{
    m_pending = pending;
}

//-----------------------------------------------------------------------------
bool
Registry::Map::Ref::tracked() const
{
    return (m_pending != -1);
}


//-----------------------------------------------------------------------------
int
Registry::Map::Ref::dec()
{
    if(m_pending > 0)
    {
        m_pending--;
    }
    else if(m_pending == 0)
    {
        // error !
    }

    return m_pending;
}

//-----------------------------------------------------------------------------
int
Registry::Map::Ref::inc(int amt)
{
    if(tracked())
    {
        m_pending+=amt;
    }
    return m_pending;
}




//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Registry::Map::Value Class
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Registry::Map::Value::Value(Data &data,
                            int refs_needed)
:m_ref(refs_needed),
 m_data(NULL)
{
    m_data = data.wrap(data.data_ptr());
}

//-----------------------------------------------------------------------------
Registry::Map::Value::~Value()
{
    // empty
    if(m_data != NULL)
    {
        delete m_data;
    }
}

//-----------------------------------------------------------------------------
Data *
Registry::Map::Value::data()
{
    return m_data;
}

//-----------------------------------------------------------------------------
void *
Registry::Map::Value::data_ptr()
{
    return m_data->data_ptr();
}


//-----------------------------------------------------------------------------
Registry::Map::Ref *
Registry::Map::Value::ref()
{
    return &m_ref;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Registry::Map::Entry Class
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------

Registry::Map::Entry::Entry(Value *value, int refs_needed)
: m_ref(refs_needed),
  m_value(value)
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Entry::~Entry()
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Value *
Registry::Map::Entry::value()
{
    return m_value;
}


//-----------------------------------------------------------------------------
Registry::Map::Ref *
Registry::Map::Entry::ref()
{
    return &m_ref;
}


//-----------------------------------------------------------------------------
Data *
Registry::Map::Entry::data()
{
    return value()->data();
}



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Registry::Map Class
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


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
void
Registry::Map::add(const std::string &key,
                   Data &data,
                   int refs_needed)
{
    // if key already exists, throw an error

    void *data_ptr = data.data_ptr();

    // check if we are already tracking this pointer
    std::map<void*,Value*>::iterator itr = m_values.find(data_ptr);
    if( itr != m_values.end() )
    {
        Value *val = itr->second;
        // if we are already tracking it, fetch the value and
        // inc the refs needed
        val->ref()->inc(refs_needed);

        // create a new entry assoced with this pointer
        Entry *ent = new Entry(val,refs_needed);
        // add to our entries
        m_entries[key] = ent;
    }
    else
    {
        // if we aren't already tracking it, we can create a
        // new value and entry
    
        Value *val = new Value(data,refs_needed);
        m_values[data_ptr] = val;

        Entry *ent = new Entry(val,refs_needed);
        m_entries[key] = ent;
    }
}

//-----------------------------------------------------------------------------
Registry::Map::Entry *
Registry::Map::fetch_entry(const std::string &key)
{
    return m_entries[key];
}

//-----------------------------------------------------------------------------
Registry::Map::Value *
Registry::Map::fetch_value(void *data_ptr)
{
    return m_values[data_ptr];
}


//-----------------------------------------------------------------------------
bool
Registry::Map::has_entry(const std::string &key)
{
    std::map<std::string,Entry*>::const_iterator itr;
    itr = m_entries.find(key);
    return itr != m_entries.end();
}

//-----------------------------------------------------------------------------
bool
Registry::Map::has_value(void *data_ptr)
{
    std::map<void*,Value*>::const_iterator itr;
    itr = m_values.find(data_ptr);
    return itr != m_values.end();
}


//-----------------------------------------------------------------------------
void
Registry::Map::dec(const std::string &key)
{

    Entry *ent   = fetch_entry(key);
    Value *value = ent->value();
    
    int ent_refs = ent->ref()->dec();
    
    if(ent_refs == 0)
    {
        ALPINE_INFO("Registry Removing: " << key);

        // clean up bookkeeping obj
        delete ent;
        m_entries.erase(key);
    }

    int val_refs = value->ref()->dec();

    if(val_refs == 0)
    {
        
        void *data_ptr = value->data_ptr();
        
        Node rel_info;
        ostringstream oss;
        oss << data_ptr;
        
        rel_info[oss.str()]["pending"] = value->ref()->pending();

        ALPINE_INFO("Registry Releasing: " << rel_info.to_json());

        value->data()->release();
                
        // clean up bookkeeping obj
        delete value;
        m_values.erase(data_ptr);
    }
}


//-----------------------------------------------------------------------------
void
Registry::Map::info(Node &out)
{
    out.reset();

    Node &ents = out["entries"];
    std::map<std::string,Entry*>::iterator eitr;
    
    for(eitr = m_entries.begin(); eitr != m_entries.end(); eitr++)
    {
        Entry *ent = eitr->second;
        ents[eitr->first]["pending"] = ent->ref()->pending();
        ent->data()->info(ents[eitr->first]["data"]);
    }

    Node &ptrs = out["pointers"];

    std::map<void*,Value*>::iterator vitr;

    ostringstream oss;
    for(vitr = m_values.begin(); vitr != m_values.end(); vitr++)
    {
        oss << vitr->first;
        Value *v= vitr->second;
        ptrs[oss.str()]["pending"] = v->ref()->pending();
        oss.str("");
    }

}

//-----------------------------------------------------------------------------
void
Registry::Map::reset()
{
    // release anything still tracked
    std::map<void*,Value*>::iterator vitr;
    for(vitr = m_values.begin(); vitr != m_values.end(); vitr++)
    {
        Value *v = vitr->second;
        if(v->ref()->tracked())
        {
            v->data()->release();
        }
    }


    // clean up internally allcoed stuff


    std::map<std::string,Entry*>::iterator eitr;
    for(eitr = m_entries.begin(); eitr != m_entries.end(); eitr++)
    {
        Entry *e = eitr->second;
        delete e;
    }
    m_entries.clear();

    
    // std::map<void*,Value*>::iterator vitr;
    for(vitr = m_values.begin(); vitr != m_values.end(); vitr++)
    {
        Value *v = vitr->second;
        delete v;
    }

    m_values.clear();
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
    reset();
    delete m_map;
}

//-----------------------------------------------------------------------------
bool 
Registry::has_entry(const std::string &key)
{
    return m_map->has_entry(key);
}

//-----------------------------------------------------------------------------
void
Registry::consume(const std::string &key)
{
    if(m_map->has_entry(key))
    {
        m_map->dec(key);
    }
}


//-----------------------------------------------------------------------------
void
Registry::reset()
{
    m_map->reset();
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
// private helper
    
//-----------------------------------------------------------------------------
Data &
Registry::fetch_data(const std::string &key)
{
    if(!m_map->has_entry(key))
    {
        ALPINE_ERROR("Attempt to fetch unknown key: " << key);
    }
    
    return *m_map->fetch_entry(key)->value()->data();
}

//-----------------------------------------------------------------------------
void
Registry::add_entry(const std::string &key,
                    Data &data,
                    int refs_needed)
{
    if(m_map->has_entry(key))
    {
        ALPINE_WARN("Attempt to overwrite existing entry with key: " << key);
    }
    else
    {
        m_map->add(key, data, refs_needed);
    }
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



