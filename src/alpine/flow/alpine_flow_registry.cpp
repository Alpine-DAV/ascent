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
             Ref();
            ~Ref();

             Ref(const Ref &r);
             Ref &operator=(Ref &v);

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
            Value();
            ~Value();

            Value(Value &v);
            Value &operator=(Value &v);

            Data     &data();
            void      set_data(Data &data);
            Ref      &ref();
 
            void     *data_ptr();
 
        private:
            Ref      m_ref;
            Data     m_data;

        
    };

    class Entry
    {
        public:
                 Entry();
                 Entry(Entry &ent);
                 Entry &operator=(Entry &ent);
                 
                 ~Entry();

                 Value &value();
                 void   set_value_ptr(Value *v);
                 Data  &data();
                 Ref   &ref();

        private:
            Ref    m_ref;
            Value *m_value;

        
    };

 
    Map();
   ~Map();

    void   add(const std::string &key,
               Data &data,
               int refs_needed=-1);

    bool   has_entry(const std::string &key);
    bool   has_value(void *data_ptr);
        
    Entry &fetch_entry(const std::string &key);
    
    Value &fetch_value(void *data_ptr);
    
    void   dec(const std::string &key);
    
    void   info(Node &out);
    
    void   reset();
    
private:

    std::map<void*,Value>         m_values;
    std::map<std::string,Entry>   m_entries;

};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Registry::Map::Ref Class
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Registry::Map::Ref::Ref()
:m_pending(-1)
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Ref::~Ref()
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Ref::Ref(const Ref &r)
:m_pending(r.m_pending)
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Ref &
Registry::Map::Ref::operator=(Registry::Map::Ref &r)
{
    if(&r != this)
    {
        m_pending = r.m_pending;
    }
    return *this;
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
Registry::Map::Value::Value()
:m_ref(),
 m_data()
{
    // empty
}


//-----------------------------------------------------------------------------
Registry::Map::Value::Value(Value &v)
:m_ref(v.m_ref),
 m_data(v.m_data)
{
    // empty
}


//-----------------------------------------------------------------------------
Registry::Map::Value &
Registry::Map::Value::operator=(Registry::Map::Value &v)
{
    if(&v != this)
    {
        m_ref  = v.m_ref;
        m_data = v.m_data;
    }
    return *this;
}


//-----------------------------------------------------------------------------
Registry::Map::Value::~Value()
{
    // empty
}

//-----------------------------------------------------------------------------
Data &
Registry::Map::Value::data()
{
    return m_data;
}

//-----------------------------------------------------------------------------
void *
Registry::Map::Value::data_ptr()
{
    return m_data.data_ptr();
}

//-----------------------------------------------------------------------------
void
Registry::Map::Value::set_data(Data &data)
{
    m_data = data;
}


//-----------------------------------------------------------------------------
Registry::Map::Ref &
Registry::Map::Value::ref()
{
    return m_ref;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Registry::Map::Entry Class
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------

Registry::Map::Entry::Entry()
: m_ref(),
  m_value(NULL)
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Entry::Entry(Entry &e)
:m_ref(e.m_ref),
 m_value(e.m_value)
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Entry &
Registry::Map::Entry::operator=(Registry::Map::Entry &e)
{
    if(&e != this)
    {
        m_ref   = e.m_ref;
        m_value = e.m_value;
    }
    return *this;
}


//-----------------------------------------------------------------------------
Registry::Map::Entry::~Entry()
{
    // empty
}

//-----------------------------------------------------------------------------
Registry::Map::Value &
Registry::Map::Entry::value()
{
    return *m_value;
}


//-----------------------------------------------------------------------------
void
Registry::Map::Entry::set_value_ptr(Registry::Map::Value *v)
{
    m_value = v;
}



//-----------------------------------------------------------------------------
Registry::Map::Ref &
Registry::Map::Entry::ref()
{
    return m_ref;
}


//-----------------------------------------------------------------------------
Data &
Registry::Map::Entry::data()
{
    return value().data();
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
    std::map<void*,Value>::iterator itr = m_values.find(data_ptr);
    if( itr != m_values.end() )
    {
        // if we are already tracking it, fetch the value and
        // inc the refs needed
        itr->second.ref().inc(refs_needed);

        Entry &ent = m_entries[key];
        // wire up the entry
        ent.set_value_ptr(&itr->second);
        ent.ref().set_pending(refs_needed);

    }
    else
    {
        // if we aren't already tracking it, we can create a
        // new value and entry
    
        Value &val = m_values[data_ptr];
        val.set_data(data);
        val.ref().set_pending(refs_needed);
    
    
        Entry &ent = m_entries[key];
        // wire up the entry
        ent.set_value_ptr(&m_values[data_ptr]);
        ent.ref().set_pending(refs_needed);
    }
}

//-----------------------------------------------------------------------------
Registry::Map::Entry &
Registry::Map::fetch_entry(const std::string &key)
{
    return m_entries[key];
}

//-----------------------------------------------------------------------------
Registry::Map::Value &
Registry::Map::fetch_value(void *data_ptr)
{
    return m_values[data_ptr];
}


//-----------------------------------------------------------------------------
bool
Registry::Map::has_entry(const std::string &key)
{
    std::map<std::string,Entry>::const_iterator itr;
    itr = m_entries.find(key);
    return itr != m_entries.end();
}

//-----------------------------------------------------------------------------
bool
Registry::Map::has_value(void *data_ptr)
{
    std::map<void*,Value>::const_iterator itr;
    itr = m_values.find(data_ptr);
    return itr != m_values.end();
}


//-----------------------------------------------------------------------------
void
Registry::Map::dec(const std::string &key)
{

    Entry &ent = m_entries[key];

    Value  &value = ent.value();
    void *data_ptr = value.data_ptr();
    
    int ent_refs = ent.ref().dec();
    
    if(ent_refs == 0)
    {
        ALPINE_INFO("Registry Removing: " << key);
        m_entries.erase(key);
    }

    int val_refs = value.ref().dec();

    if(val_refs == 0)
    {
        Node rel_info;
        ostringstream oss;
        oss << data_ptr;
        
        rel_info[oss.str()]["pending"] = value.ref().pending();
        ALPINE_INFO("Registry Releasing: " << rel_info.to_json());
                
        m_values[data_ptr].data().release();
        m_values.erase(data_ptr);
    }
}


//-----------------------------------------------------------------------------
void
Registry::Map::info(Node &out)
{
    out.reset();

    Node &ents = out["entries"];
    std::map<std::string,Entry>::iterator eitr;
    
    for(eitr = m_entries.begin(); eitr != m_entries.end(); eitr++)
    {
        Entry &ent = eitr->second;
        ents[eitr->first]["pending"] = ent.ref().pending();
        ent.data().info(ents[eitr->first]["data"]);
    }

    Node &ptrs = out["pointers"];

    std::map<void*,Value>::iterator vitr;

    ostringstream oss;
    for(vitr = m_values.begin(); vitr != m_values.end(); vitr++)
    {
        oss << vitr->first;
        Value &v= vitr->second;
        ptrs[oss.str()]["pending"] = v.ref().pending();
        oss.str("");
    }

}

//-----------------------------------------------------------------------------
void
Registry::Map::reset()
{
    m_entries.clear();
    
    std::map<void*,Value>::iterator vitr;
    for(vitr = m_values.begin(); vitr != m_values.end(); vitr++)
    {
        Value &v = vitr->second;
        if(v.ref().tracked())
        {
            v.data().release();
        }
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
void
Registry::add(const std::string &key, 
              Data data,
              int ref_count)
{
    
    m_map->add(key,data,ref_count);
}

//-----------------------------------------------------------------------------
bool 
Registry::has_entry(const std::string &key)
{
    return m_map->has_entry(key);
}

//-----------------------------------------------------------------------------
Data 
Registry::fetch(const std::string &key)
{
    if(!m_map->has_entry(key))
    {
        ALPINE_WARN("Attempt to fetch unknown key: " << key);
        return Data();
    }
    else
    {
        return m_map->fetch_entry(key).value().data();
    }
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
};
//-----------------------------------------------------------------------------
// -- end alpine::flow --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------



