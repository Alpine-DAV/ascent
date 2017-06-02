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
/// file: alpine_flow_registry.hpp
///
//-----------------------------------------------------------------------------

#ifndef ALPINE_FLOW_REGISTRY_HPP
#define ALPINE_FLOW_REGISTRY_HPP

#include <conduit.hpp>

#include <alpine_flow_data.hpp>


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


// registry use cases:
    
// data into and out of the data flow network:
// 
// aliasing a registry input 
// providing a registry output 
    

// data created and consumed within a data flow network
// data aliased within a data flow network
    

// goals:
// we want to be able to track data used in the network and free it
//
//
// all internal connections for the network are tracked.
// each output has a unique key, and we expect to know the number
// of refs needed for each key
//
// that said keys can be associated with aliased pointers.
//
// r.add_entry("src",ptr,-1); // keep forever
//
// // output of filter a simply passes the pointer thru w/
// // a diff ref count ...
//
// r.add_entry("a",ptr,1)
//
// // When a is consumed, we want to remove the entry for a, but we don't
// // want to delete the pointer.
// // vs:
// //
// r.add_entry("b",new_ptr,1)
// // for this case we want to clean up the new pointer.
//
// // todo this, we need to track both how the pointers are used
// // and track keys


// filter exe:
//
// T * my_data_1 = input(0)->data();
// T * my_data_2 = input(1)->data();
// aliased! 
// filters that consume output will each inc refs_need of my_data_2 by 1
// output()->set(my_data_2)



// filter exe:
//
// T * my_data_1 = input(0)->data();
// T * my_data_2 = input(1)->data();
// life will be managed by the registry
// output()->set(my_new_data)
    
    
// RegistrySource filter:
// hoists a registry entry into the data flow
// expects refs_needed to be to -1 (not-managed)
// output of the filter will alias an existing entry
// can add as refs_need -1


// RegistrySink filter:
// hoists a result into a registry entry that has refs_needed
// equal -1 (not-managed)
// output of the filter will alias an existing entry

//-----------------------------------------------------------------------------
class Registry
{
public:

    // Creation and Destruction
    Registry();
   ~Registry();
    
    void        add(const std::string &key, 
                    Data data,
                    int refs_needed=-1); // -1 means don't manage mem
    
    /// fetch entry by key, does not dec
    Data        fetch(const std::string &key);
    
    void        consume(const std::string &key);
    
    void        info(conduit::Node &out);
    std::string to_json();
    void        print();

private:
    // private class that hides imp from main interface
    class Map;
    Map *m_map;

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


