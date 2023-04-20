//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_builtin_filters.hpp
///
//-----------------------------------------------------------------------------

#ifndef FLOW_BUILTIN_FILTERS_HPP
#define FLOW_BUILTIN_FILTERS_HPP

#include <flow_filter.hpp>


//-----------------------------------------------------------------------------
// -- begin flow:: --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
// -- begin flow::filters --
//-----------------------------------------------------------------------------
namespace filters
{

//-----------------------------------------------------------------------------
///
/// Alias returns its input as output.
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class FLOW_API Alias : public ::flow::Filter
{
public:
    Alias();
   ~Alias();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
///
/// Depennt Alias returns its input as output with a dummy connection
/// to enforce and ordering.
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class FLOW_API DependentAlias : public ::flow::Filter
{
public:
    DependentAlias();
   ~DependentAlias();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};


//-----------------------------------------------------------------------------
///
/// Filters Related to Registry Access
///
//-----------------------------------------------------------------------------

/// RegistrySource filter:
/// hoists a registry entry into the data flow
/// expects refs_needed to be to -1 (not-managed)
/// output of the filter will alias an existing entry

//-----------------------------------------------------------------------------
class FLOW_API RegistrySource : public Filter
{
public:
    RegistrySource();
   ~RegistrySource();

    virtual void   declare_interface(conduit::Node &i);
    virtual void   execute();
};

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end flow:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


