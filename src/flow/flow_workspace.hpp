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
/// file: flow_workspace.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_WORKSPACE_HPP
#define FLOW_WORKSPACE_HPP

#include <conduit.hpp>

#include <flow_data.hpp>
#include <flow_registry.hpp>
#include <flow_graph.hpp>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
///
/// Workspace 
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class Workspace
{
public:

    friend class Graph;

   // ------------------------------------------------------------------------
   /// Workspace instance methods
   // ------------------------------------------------------------------------

    Workspace();
   ~Workspace();
   
    /// access the filter graph
    Graph           &graph();
    /// const access to the filter graph
    const Graph     &graph() const;
   
    /// access the registry
    Registry        &registry();
    /// const access to the registry
    const Registry  &registry() const;
   
    /// compute and return the graph traverals 
    void             traversals(conduit::Node &out);
   
    /// execute the filter graph.
    void             execute();
    
    /// reset the registry and graph
    void             reset();
   
    /// create human understandable tree that describes the state
    /// of the workspace
    void           info(conduit::Node &out) const;
    /// create json string from info
    std::string    to_json() const;
    /// print json version of info
    void           print() const;

    // ------------------------------------------------------------------------
    /// Interface to set and obtain the MPI communicator.
    ///
    /// We use an integer handle from MPI_Comm_c2f to avoid
    /// a header dependency of mpi just for the handle. 
    ///
    // ------------------------------------------------------------------------
    void static set_default_mpi_comm(int mpi_comm_id);
    int  static default_mpi_comm();

    // ------------------------------------------------------------------------
    /// filter factory interface
    // ------------------------------------------------------------------------

    /// register a new type 
    static void register_filter_type(FilterFactoryMethod fr);
    /// check if type with given name is registered
    static bool supports_filter_type(const std::string &filter_type);
    /// check if type with given factory is registered
    static bool supports_filter_type(FilterFactoryMethod fr);
    
    /// remove type with given name if registered
    static void remove_filter_type(const std::string &filter_type);
    /// remove all registered types
    static void clear_supported_filter_types();

    /// helper to for registering a filter type that does not provide its own
    /// FilterFactoryMethod
    template <class T>
    static void register_filter_type()
    {
        register_filter_type(&CreateFilter<T>);
    }

    /// helper for checkeding if a filter type is registered
    template <class T>
    static bool supports_filter_type()
    {
        return supports_filter_type(&CreateFilter<T>);
    }


private:

    static Filter *create_filter(const std::string &filter_type);

    static int  m_default_mpi_comm;

    class ExecutionPlan;
    class FilterFactory;

    Graph       m_graph;
    Registry    m_registry;
   

   
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


