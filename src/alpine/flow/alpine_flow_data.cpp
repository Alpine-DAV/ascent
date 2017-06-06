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
/// file: alpine_flow_data.cpp
///
//-----------------------------------------------------------------------------

#include "alpine_flow_data.hpp"

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>



//-----------------------------------------------------------------------------
// alpine includes
//-----------------------------------------------------------------------------
#include <alpine_logging.hpp>

using namespace conduit;
using namespace std;

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
//-----------------------------------------------------------------------------
//
// Data
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
Data::Data()
: m_data_ptr(NULL)
{
    
}

//-----------------------------------------------------------------------------
Data::Data(conduit::Node *node)
: m_data_ptr(NULL)
{
    set(node);
}

//-----------------------------------------------------------------------------
Data::Data(Data &ds)
: m_data_ptr(NULL)
{
    // shallow cpy
    m_data_ptr = ds.m_data_ptr;
}

//-----------------------------------------------------------------------------
Data &
Data::operator=(Data &v)
{
    if(&v != this)
    {
        m_data_ptr = v.m_data_ptr;
    }
    return *this;
}

//-----------------------------------------------------------------------------
Data &
Data::operator=(const Data &v)
{
    if(&v != this)
    {
        m_data_ptr = v.m_data_ptr;
    }
    return *this;
}


//-----------------------------------------------------------------------------
Data::~Data()
{

}


//-----------------------------------------------------------------------------
void
Data::set(Node *data_ptr)
{
    m_data_ptr = data_ptr;
}

//-----------------------------------------------------------------------------
void
Data::release()
{
    if(m_data_ptr != NULL)
    {
        delete m_data_ptr;
    }
}

//-----------------------------------------------------------------------------
void *
Data::data_ptr()
{
    return (void*)m_data_ptr;
}

//-----------------------------------------------------------------------------
conduit::Node *
Data::as_node_ptr()
{
    return m_data_ptr;
}

//-----------------------------------------------------------------------------
Data::operator Node &()
{
    Node *node_ptr = as_node_ptr();
    if(node_ptr == NULL)
    {
        ALPINE_ERROR("Data is not a conduit::Node instance");
    }

    return *as_node_ptr();
}

//-----------------------------------------------------------------------------
Data::operator Node *()
{
    return as_node_ptr();
}

//-----------------------------------------------------------------------------
void
Data::info(Node &out)
{
    out.reset();
    ostringstream oss;
    oss << m_data_ptr;
    out["node_ptr"] = oss.str();
}



//-----------------------------------------------------------------------------
std::string
Data::to_json()
{
    Node out;
    info(out);
    ostringstream oss;
    out.to_json_stream(oss);
    return oss.str();
}

//-----------------------------------------------------------------------------
void
Data::print()
{
    ALPINE_INFO(to_json());
}


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



