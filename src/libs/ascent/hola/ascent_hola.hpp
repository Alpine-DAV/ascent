//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_hola.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_HOLA_HPP
#define ASCENT_HOLA_HPP

#include <ascent_config.h>
#include <ascent_exports.h>

#include <string>
#include <conduit.hpp>


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{
//-----------------------------------------------------------------------------
// Hola is a way to say hello again to data captured using Ascent extracts.
//-----------------------------------------------------------------------------
void ASCENT_API hola(const std::string &source,
                     const conduit::Node &options,
                     conduit::Node &data);

};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------

