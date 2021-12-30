//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: about.cpp
///
//-----------------------------------------------------------------------------
#include <ascent.hpp>

#ifdef ABOUT_MPI
#include <mpi.h>
#endif

int main (int argc, char *argv[])
{
    int par_size = 1;
    int par_rank = 0;

#ifdef ABOUT_MPI
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
#endif

    if(par_rank == 0)
    {
        conduit::Node info;
        ascent::about(info);
        info.print();
    }

#ifdef ABOUT_MPI
  MPI_Finalize();
#endif
  return 0;
}
