//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_registry.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_REGISTRY_HPP
#define FLOW_REGISTRY_HPP

#include <conduit.hpp>

#include <flow_exports.h>
#include <flow_config.h>
#include <flow_data.hpp>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{


//-----------------------------------------------------------------------------
///
/// Registry
///
//-----------------------------------------------------------------------------



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
// T * my_data_1 = input(0);
// T * my_data_2 = input(1);
// aliased!
// filters that consume output will each inc refs_need of my_data_2 by 1
// output()->set(my_data_2)


// filter exe:
//
// T * my_data_1 = input(0);
// T * my_data_2 = input(1);
// life will be managed by the registry
// output()->set(my_new_data)

//-----------------------------------------------------------------------------
class FLOW_API Registry
{
public:

    // Creation and Destruction
    Registry();
   ~Registry();


    /// adds a new entry to the registry
    /// if refs_needed == -1, the entry is not tracked
    template <class T>
    void add(const std::string &key,
             T *data_ptr,
             int refs_needed=-1) // -1 means don't track and release mem
    {
        DataWrapper<T> data(data_ptr);
        add(key,data,refs_needed);
    }

    /// fetch entry by key
    /// (does not decrement refs_needed)
    template <class T>
    T *fetch(const std::string &key)
    {
        return fetch(key).value<T>();
    }

    /// adds a new entry to the registry
    /// if refs_needed == -1, the entry is not tracked
    void           add(const std::string &key,
                       Data &data,
                       int refs_needed);

    /// fetch entry by key, does not decrement refs_needed
    Data          &fetch(const std::string &key);

    /// check if the registry contains entry with given name
    bool           has_entry(const std::string &key);

    /// decrement refs needed if entry is tracked,
    ///  if refs_needed = 0 releases the data held by the entry.
    void           consume(const std::string &key);

    /// removes entry from that data store w/o releasing data.
    void           detach(const std::string &key);

    /// clears registry entries and releases any outstanding
    /// tracked data refs.
    void           reset();

    /// create human understandable tree that describes the state
    /// of the registry
    void           info(conduit::Node &out) const;
    /// create json string from info
    std::string    to_json() const;
    /// create yaml string from info
    std::string    to_yaml() const;
    /// print yaml version of info
    void           print() const;


private:

    // internal private class that hides imp from main interface
    class Map;
    Map *m_map;
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


