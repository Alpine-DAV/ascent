//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
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



