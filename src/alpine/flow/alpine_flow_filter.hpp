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
/// file: alpine_flow_filter.hpp
///
//-----------------------------------------------------------------------------

#ifndef ALPINE_FLOW_FILTER_HPP
#define ALPINE_FLOW_FILTER_HPP

#include <conduit.hpp>

#include <alpine_flow_data.hpp>
#include <alpine_flow_registry.hpp>


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


class Workspace;
class Graph;

//-----------------------------------------------------------------------------
class Filter
{
public:
    
    friend class Graph;
    friend class Workspace;
    
    virtual ~Filter();

    // filter properties 
    std::string           name();
    std::string           type_name();
    const conduit::Node  &port_names();
    const conduit::Node  &default_params();
    bool                  output_port();

    bool                  has_port(const std::string &name);

    // imp to do work in subclass
    virtual void          execute() = 0;

    virtual bool          verify_params(const conduit::Node &params,
                                        conduit::Node &info);

    // methods used to implement filter exe

    conduit::Node         &params();
    conduit::Node         &properties();

    Data                  &input(const std::string &port_name);
    Data                  &output();
    
    Graph                 &graph();
    
    // graph().connect(f->name(),this->name(),port_name);
    void                  connect_input_port(const std::string &port_name,
                                             Filter *filter);

    
    /// human friendly output
    void                   info(conduit::Node &out);
    std::string            to_json();
    void                   print();

protected:
    Filter();

private:

    // used by ws interface prior to imp exec
    void                    set_input(const std::string &port_name,
                                      Data ds);

    void                    init(Graph *graph,
                                 const std::string &name,
                                 const conduit::Node &params);

    void                    reset_inputs_and_output();


    Graph                         *m_graph;
    conduit::Node                  m_props;
    Data                           m_out;
    std::map<std::string,Data>     m_inputs;

};

//-----------------------------------------------------------------------------
typedef Filter *(*FilterType)();


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


