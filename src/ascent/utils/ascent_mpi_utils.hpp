//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_utils.hpp
///
//-----------------------------------------------------------------------------
#ifndef ASCENT_MPI_UTILS_HPP
#define ASCENT_MPI_UTILS_HPP

#include <set>
#include <string>
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//
// returns true if all ranks say true
//
bool global_agreement(bool vote);

//
// returns true if any ranks says true
//
bool global_someone_agrees(bool vote);

//
// gathers strings from all ranks
//
void gather_strings(std::set<std::string> &set);

int mpi_rank();
int mpi_size();

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------


#endif
//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


