//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-716457
// 
// All rights reserved.
// 
// This file is part of Strawman. 
// 
// For details, see: http://software.llnl.gov/strawman/.
// 
// Please also read strawman/LICENSE
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
/// file: strawman_c.cpp
///
//-----------------------------------------------------------------------------

#include "strawman.h"

#include <conduit.hpp>
#include <conduit_cpp_to_c.hpp>

#include "strawman.hpp"
#include "strawman_block_timer.hpp"
using conduit::Node;

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {


//---------------------------------------------------------------------------//
strawman::Strawman *
cpp_strawman(Strawman *v)
{
    return static_cast<strawman::Strawman*>(v);
}

//---------------------------------------------------------------------------//
Strawman *
c_strawman(strawman::Strawman *v)
{
    return (void*)v;
}

//---------------------------------------------------------------------------//
void
strawman_about(conduit_node *result)
{
    Node &n = conduit::cpp_node_ref(result);
    strawman::about(n);
}

//---------------------------------------------------------------------------//
Strawman *
strawman_create()
{
    return c_strawman(new strawman::Strawman());
}

//---------------------------------------------------------------------------//
void
strawman_open(Strawman *c_sman,
              conduit_node *c_options)
{
    strawman::Strawman *v = cpp_strawman(c_sman);
    Node  *n = static_cast<Node*>(c_options);
    v->Open(*n);
}

//---------------------------------------------------------------------------//
void
strawman_publish(Strawman *c_sman,
                 conduit_node *c_data)
{
    strawman::Strawman *v = cpp_strawman(c_sman);
    Node  *n = static_cast<Node*>(c_data);
    v->Publish(*n);
}

//---------------------------------------------------------------------------//
void
strawman_execute(Strawman *c_sman,
                 conduit_node *c_actions)
{
    strawman::Strawman *v = cpp_strawman(c_sman);
    Node  *n = static_cast<Node*>(c_actions);
    v->Execute(*n);
}

//---------------------------------------------------------------------------//
void
strawman_close(Strawman *c_sman)
{
    strawman::Strawman *v = cpp_strawman(c_sman);
    v->Close();
}

//---------------------------------------------------------------------------//
void
strawman_destroy(Strawman *c_sman)
{
   strawman::Strawman *v = cpp_strawman(c_sman);
    delete v;
}

void strawman_timer_start(char *name)
{
  strawman::BlockTimer::StartTimer(name); 
}

//---------------------------------------------------------------------------//
void strawman_timer_stop(char *name)
{
  strawman::BlockTimer::StopTimer(name); 
}

//---------------------------------------------------------------------------//
void strawman_timer_write()
{
  strawman::BlockTimer::WriteLogFile(); 
}
}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------
