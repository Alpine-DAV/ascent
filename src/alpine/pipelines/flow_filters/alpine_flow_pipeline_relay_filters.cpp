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
/// file: alpine_flow_pipeline_relay_filters.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_flow_pipeline_relay_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_relay.hpp>

//-----------------------------------------------------------------------------
// alpine includes
//-----------------------------------------------------------------------------
#include <alpine_logging.hpp>
#include <alpine_flow_graph.hpp>
#include <alpine_flow_workspace.hpp>

using namespace conduit;
using namespace std;

using namespace alpine::flow;

//-----------------------------------------------------------------------------
// -- begin alpine:: --
//-----------------------------------------------------------------------------
namespace alpine
{

//-----------------------------------------------------------------------------
// -- begin alpine::pipeline --
//-----------------------------------------------------------------------------
namespace pipeline
{

//-----------------------------------------------------------------------------
// -- begin alpine::pipeline::flow --
//-----------------------------------------------------------------------------
namespace flow
{

//-----------------------------------------------------------------------------
// -- begin alpine::pipeline::flow::filters --
//-----------------------------------------------------------------------------
namespace filters
{



//-----------------------------------------------------------------------------
// helper shared by io save and load
//-----------------------------------------------------------------------------
bool
verify_io_params(const conduit::Node &params,
                 conduit::Node &info)
{
    bool res = true;
    
    if( !params.has_child("path") ) 
    {
        info["errors"].append() = "missing required entry 'path'";
        res = false;
    }
    else if(!params["path"].dtype().is_string())
    {
        info["errors"].append() = "'path' must be a string";
        res = false;
    }
    else if(params["path"].as_string().empty())
    {
        info["errors"].append() = "'path' is an empty string";
        res = false;
    }
    
    
    if( params.has_child("protocol") ) 
    {
        if(!params["protocol"].dtype().is_string())
        {
            info["errors"].append() = "optional entry 'protocol' must be a string";
            res = false;
        }
        else if(params["protocol"].as_string().empty())
        {
            info["errors"].append() = "'protocol' is an empty string";
            res = false;
        }
        else
        {
            info["info"].append() = "includes 'protocol'";
        }
    }

    return res;
}


//-----------------------------------------------------------------------------
RelayIOSave::RelayIOSave()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RelayIOSave::~RelayIOSave()
{
// empty
}

//-----------------------------------------------------------------------------
void 
RelayIOSave::declare_interface(Node &i)
{
    i["type_name"]   = "relay_io_save";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
RelayIOSave::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    return verify_io_params(params,info);
}


//-----------------------------------------------------------------------------
void 
RelayIOSave::execute()
{
    std::string path, protocol;
    path = params()["path"].as_string();
    
    // TODO check if we need to expand the path (MPI) case
    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    if(!input("in").check_type<Node>())
    {
        // error
    }
    
    Node *in = input<Node>("in");
    
    if(protocol.empty())
    {
        conduit::relay::io::save(*in,path);
    }
    else
    {
        conduit::relay::io::save(*in,path,protocol);
    }

}



//-----------------------------------------------------------------------------
RelayIOLoad::RelayIOLoad()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
RelayIOLoad::~RelayIOLoad()
{
// empty
}

//-----------------------------------------------------------------------------
void 
RelayIOLoad::declare_interface(Node &i)
{
    i["type_name"]   = "relay_io_load";
    i["port_names"] = DataType::empty();
    i["output_port"] = "true";
}

//-----------------------------------------------------------------------------
bool
RelayIOLoad::verify_params(const conduit::Node &params,
                           conduit::Node &info)
{
    return verify_io_params(params,info);
}


//-----------------------------------------------------------------------------
void 
RelayIOLoad::execute()
{
    std::string path, protocol;
    path = params()["path"].as_string();
    
    // TODO check if we need to expand the path (MPI) case
    if(params().has_child("protocol"))
    {
        protocol = params()["protocol"].as_string();
    }

    Node *res = new Node();
    
    if(protocol.empty())
    {
        conduit::relay::io::load(path,*res);
    }
    else
    {
        conduit::relay::io::load(path,protocol,*res);
    }
    
    set_output<Node>(res);

}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine::pipeline::flow::filters --
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine::pipeline::flow --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine::pipeline --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end alpine:: --
//-----------------------------------------------------------------------------





