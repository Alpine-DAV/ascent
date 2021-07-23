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
/// file: ascent_c.cpp
///
//-----------------------------------------------------------------------------

#include "ascent.h"

#include <conduit.hpp>
#include <conduit_cpp_to_c.hpp>

#include "ascent.hpp"
#include "ascent_block_timer.hpp"
using conduit::Node;

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {


//---------------------------------------------------------------------------//
ascent::Ascent *
cpp_ascent(Ascent *v)
{
    return static_cast<ascent::Ascent*>(v);
}

//---------------------------------------------------------------------------//
Ascent *
c_ascent(ascent::Ascent *v)
{
    return (void*)v;
}

//---------------------------------------------------------------------------//
void
ascent_about(conduit_node *result)
{
    Node &n = conduit::cpp_node_ref(result);
    ascent::about(n);
}

//---------------------------------------------------------------------------//
Ascent *
ascent_create()
{
    return c_ascent(new ascent::Ascent());
}

//---------------------------------------------------------------------------//
void
ascent_open(Ascent *c_ascent,
            conduit_node *c_options)
{
    ascent::Ascent *v = cpp_ascent(c_ascent);
    Node &n = conduit::cpp_node_ref(c_options);
    v->open(n);
}

//---------------------------------------------------------------------------//
void
ascent_publish(Ascent *c_ascent,
               conduit_node *c_data)
{
    ascent::Ascent *v = cpp_ascent(c_ascent);
    Node &n = conduit::cpp_node_ref(c_data);
    v->publish(n);
}

//---------------------------------------------------------------------------//
void
ascent_execute(Ascent *c_ascent,
               conduit_node *c_actions)
{
    ascent::Ascent *v = cpp_ascent(c_ascent);
    Node &n = conduit::cpp_node_ref(c_actions);
    v->execute(n);
}

//---------------------------------------------------------------------------//
void
ascent_info(Ascent *c_ascent,
            conduit_node *c_out)
{
    ascent::Ascent *v = cpp_ascent(c_ascent);
    Node &n = conduit::cpp_node_ref(c_out);
    v->info(n);
}

//---------------------------------------------------------------------------//
void
ascent_close(Ascent *c_ascent)
{
    ascent::Ascent *v = cpp_ascent(c_ascent);
    v->close();
}

//---------------------------------------------------------------------------//
void
ascent_destroy(Ascent *c_ascent)
{
    ascent::Ascent *v = cpp_ascent(c_ascent);
    delete v;
}

//---------------------------------------------------------------------------//
void
ascent_timer_start(char *name)
{
    ascent::BlockTimer::StartTimer(name);
}

//---------------------------------------------------------------------------//
void
ascent_timer_stop(char *name)
{
    ascent::BlockTimer::StopTimer(name);
}

//---------------------------------------------------------------------------//
void
ascent_timer_write()
{
    ascent::BlockTimer::WriteLogFile();
}

}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------
