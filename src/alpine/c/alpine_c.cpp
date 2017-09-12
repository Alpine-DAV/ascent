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
/// file: alpine_c.cpp
///
//-----------------------------------------------------------------------------

#include "alpine.h"

#include <conduit.hpp>
#include <conduit_cpp_to_c.hpp>

#include "alpine.hpp"
#include "alpine_block_timer.hpp"
using conduit::Node;

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {


//---------------------------------------------------------------------------//
alpine::Alpine *
cpp_alpine(Alpine *v)
{
    return static_cast<alpine::Alpine*>(v);
}

//---------------------------------------------------------------------------//
Alpine *
c_alpine(alpine::Alpine *v)
{
    return (void*)v;
}

//---------------------------------------------------------------------------//
void
alpine_about(conduit_node *result)
{
    Node &n = conduit::cpp_node_ref(result);
    alpine::about(n);
}

//---------------------------------------------------------------------------//
Alpine *
alpine_create()
{
    return c_alpine(new alpine::Alpine());
}

//---------------------------------------------------------------------------//
void
alpine_open(Alpine *c_sman,
              conduit_node *c_options)
{
    alpine::Alpine *v = cpp_alpine(c_sman);
    Node  *n = static_cast<Node*>(c_options);
    v->open(*n);
}

//---------------------------------------------------------------------------//
void
alpine_publish(Alpine *c_sman,
                 conduit_node *c_data)
{
    alpine::Alpine *v = cpp_alpine(c_sman);
    Node  *n = static_cast<Node*>(c_data);
    v->publish(*n);
}

//---------------------------------------------------------------------------//
void
alpine_execute(Alpine *c_sman,
                 conduit_node *c_actions)
{
    alpine::Alpine *v = cpp_alpine(c_sman);
    Node  *n = static_cast<Node*>(c_actions);
    v->execute(*n);
}

//---------------------------------------------------------------------------//
void
alpine_close(Alpine *c_sman)
{
    alpine::Alpine *v = cpp_alpine(c_sman);
    v->close();
}

//---------------------------------------------------------------------------//
void
alpine_destroy(Alpine *c_sman)
{
   alpine::Alpine *v = cpp_alpine(c_sman);
    delete v;
}

void alpine_timer_start(char *name)
{
  alpine::BlockTimer::StartTimer(name); 
}

//---------------------------------------------------------------------------//
void alpine_timer_stop(char *name)
{
  alpine::BlockTimer::StopTimer(name); 
}

//---------------------------------------------------------------------------//
void alpine_timer_write()
{
  alpine::BlockTimer::WriteLogFile(); 
}
}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------
