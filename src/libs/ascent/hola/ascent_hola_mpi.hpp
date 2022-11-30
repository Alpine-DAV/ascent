//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_hola_mpi.hpp
///
//-----------------------------------------------------------------------------

#ifndef ASCENT_HOLA_MPI_HPP
#define ASCENT_HOLA_MPI_HPP

#include <ascent_config.h>
#include <ascent_exports.h>

#include <string>
#include <conduit.hpp>

#include <mpi.h>


//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

void ASCENT_API hola_mpi(const conduit::Node &options,
                         conduit::Node &data);


/// Creates maps used for book keeping to guide sending domains
/// from source to destination ranks.
void ASCENT_API hola_mpi_comm_map(const conduit::Node &data,
                                  MPI_Comm comm,
                                  const conduit::int32_array &world_to_src,
                                  const conduit::int32_array &world_to_dest,
                                  conduit::Node &res);

/// executes a send
void ASCENT_API hola_mpi_send(const conduit::Node &data,
                              MPI_Comm comm,
                              int src_idx,
                              const conduit::Node &comm_map);

/// executes a receive
void ASCENT_API hola_mpi_recv(MPI_Comm comm,
                              int dest_idx,
                              const conduit::Node &comm_map,
                              conduit::Node &data);

};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------

