//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_filters.cpp
///
//-----------------------------------------------------------------------------

#include <flow_filters.hpp>

//-----------------------------------------------------------------------------
// flow includes
//-----------------------------------------------------------------------------
#include <flow_workspace.hpp>
#include <flow_builtin_filters.hpp>

#ifdef FLOW_PYTHON_ENABLED
#include <flow_python_script_filter.hpp>
#endif

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
// init all built in filters
//-----------------------------------------------------------------------------
void
register_builtin()
{
    if(!Workspace::supports_filter_type<RegistrySource>())
    {
        Workspace::register_filter_type<RegistrySource>();
    }

    if(!Workspace::supports_filter_type<Alias>())
    {
        Workspace::register_filter_type<Alias>();
    }
    if(!Workspace::supports_filter_type<DependentAlias>())
    {
        Workspace::register_filter_type<DependentAlias>();
    }
#ifdef FLOW_PYTHON_ENABLED
    if(!Workspace::supports_filter_type<PythonScript>())
    {
        Workspace::register_filter_type<PythonScript>();
    }
#endif
}


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



