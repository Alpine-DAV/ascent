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
/// file: alpine_flow.hpp
///
//-----------------------------------------------------------------------------

#ifndef ALPINE_FLOW_HPP
#define ALPINE_FLOW_HPP

#include <conduit.hpp>


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
class Dataset
{
public:
    Dataset();
   ~Dataset();
   
   Dataset(Dataset &ds);
   Dataset &operator=(Dataset &ds);
    
    void           set(conduit::Node *ds);
    // void         set_data (VTKHDataset *ds);
    void           release();

    
    conduit::Node *as_node_ptr();

    // VTKHDataset *as_vtkh_dataset();

    operator conduit::Node *();
    
    //operator VtkHDataset *();

    void        info(conduit::Node &out);
    std::string to_json();
    void        print();

private:
    // private class that hides imp from main interface
    class Value;
    Value *m_value;
};


//-----------------------------------------------------------------------------
class Registry
{
public:
    
    // Creation and Destruction
    Registry();
   ~Registry();
    
    /// add a conduit:Node-based dataset
    void    add_entry(const std::string &key, 
                      conduit::Node *node,
                      int ref_count=-1);
    
    /// fetch entry by key
    /// does not dec ref
    Dataset &fetch_entry(const std::string &key);
    
    /// dec ref of given key
    void     dec_entry_ref_count(const std::string &key);
    
    
    void        info(conduit::Node &out);
    std::string to_json();
    void        print();

private:
    // private class that hides imp from main interface
    class Map;
    Map *m_map;

};

//-----------------------------------------------------------------------------
class Filter
{
public:
    Filter();
    virtual ~Filter();

    // these methods are used provide the filter interface

    virtual void         execute() = 0;
    
    std::string          type_name();
    const conduit::Node &input_port_names();
    const conduit::Node &default_params();
    bool                 output_port();
    

    conduit::Node          &params();
    conduit::Node          &properties();

    // methods used to implement filter exe
    Dataset                &input(const std::string &port_name);
    Dataset                &output();
    
    // todo, ambig with has name vs has actual entry ... 
    bool                    has_input_port(const std::string &name);

    // used by ws interface prior to exec
    void                    set_input(const std::string &port_name,
                                      Dataset &ds);

    /// human friendly output
    void                    info(conduit::Node &out);
    std::string             to_json();
    void                    print();

protected:
    void                    init(); // call in subclass constructor

private:

    // methods used by the workspace to setup a filter for exe
    conduit::Node                  m_props;
    Dataset                        m_out;
    std::map<std::string,Dataset>  m_inputs;

};


//-----------------------------------------------------------------------------
class FilterGraph
{
public:
    
    FilterGraph();
   ~FilterGraph();

    // register filter type by name
    // add by name w. opt params

    void add_filter(const std::string &name,
                    Filter *f);


    void connect(const std::string &src_name,
                 const std::string &des_name,
                 const std::string &port_name);

    bool has_filter(const std::string &name);

    void remove_filter(const std::string &name);

    /// human friendly output
    void        info(conduit::Node &out);
    std::string to_json();
    void        print();

private:

    conduit::Node                 m_edges;
    std::map<std::string,Filter*> m_filters;

};


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

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


