//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent.h
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_H
#define ASCENT_H

#include <conduit_node.h>
#include <ascent_exports.h>

//-----------------------------------------------------------------------------
//
// The C interface to ascent
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- typedef for ascent --
//-----------------------------------------------------------------------------

typedef void  Ascent;

//-----------------------------------------------------------------------------
// --ascent methods --
//-----------------------------------------------------------------------------

void ASCENT_API ascent_about(conduit_node *result);

Ascent ASCENT_API *ascent_create();

void ASCENT_API ascent_destroy(Ascent *c_ascent);

void ASCENT_API ascent_open(Ascent *c_ascent,  conduit_node *options);

void ASCENT_API ascent_publish(Ascent *c_ascent, conduit_node *data);

void ASCENT_API ascent_execute(Ascent *c_ascent, conduit_node *actions);

void ASCENT_API ascent_info(Ascent *c_ascent, conduit_node *result);

void ASCENT_API ascent_close(Ascent *c_ascent);

void ASCENT_API ascent_timer_start(char *name);

void ASCENT_API ascent_timer_stop(char *name);

void ASCENT_API ascent_timer_write();


#ifdef __cplusplus
}
#endif

//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------



