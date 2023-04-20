//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: flow_builtin_filters.cpp
///
//-----------------------------------------------------------------------------

#include "flow_builtin_filters.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>


//-----------------------------------------------------------------------------
// flow includes
//-----------------------------------------------------------------------------
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

using namespace conduit;
using namespace std;

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
Alias::Alias()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
Alias::~Alias()
{
// empty
}

//-----------------------------------------------------------------------------
void
Alias::declare_interface(Node &i)
{
    i["type_name"]   = "alias";
    i["port_names"].append() = "in";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
Alias::execute()
{
    set_output(input(0));
}

//-----------------------------------------------------------------------------
DependentAlias::DependentAlias()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DependentAlias::~DependentAlias()
{
// empty
}

//-----------------------------------------------------------------------------
void
DependentAlias::declare_interface(Node &i)
{
    i["type_name"]   = "dependent_alias";
    i["port_names"].append() = "in";
    i["port_names"].append() = "dummy";
    i["output_port"] = "true";
}


//-----------------------------------------------------------------------------
void
DependentAlias::execute()
{
    set_output(input(0));
}


//-----------------------------------------------------------------------------
RegistrySource::RegistrySource()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RegistrySource::~RegistrySource()
{
// empty
}

//-----------------------------------------------------------------------------
void
RegistrySource::declare_interface(Node &i)
{
    i["type_name"]   = "registry_source";
    i["port_names"]  = DataType::empty();
    i["output_port"] = "true";
    i["default_params"]["entry"] = "";
}


//-----------------------------------------------------------------------------
void
RegistrySource::execute()
{
    std::string key = params()["entry"].as_string();

    set_output(graph().workspace().registry().fetch(key));
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



