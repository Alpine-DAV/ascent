//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
