//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "apcomp.hpp"
#include "error.hpp"
#include <sstream>

#ifdef APCOMP_PARALLEL
#include <mpi.h>
#endif

namespace apcomp
{

static int g_mpi_comm_id = -1;


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
#ifdef APCOMP_PARALLEL // mpi case
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
check_comm()
{
  if(g_mpi_comm_id == -1)
  {
    std::stringstream msg;
    msg<<"APComp internal error. There is no valid MPI comm available. ";
    msg<<"It is likely that apcomp::mpi_comm(int) was not called.";
    throw Error(msg.str());
  }
}

//---------------------------------------------------------------------------//
void
mpi_comm(int mpi_comm_id)
{
  g_mpi_comm_id = mpi_comm_id;
}

//---------------------------------------------------------------------------//
int
mpi_comm()
{
  check_comm();
  return g_mpi_comm_id;
}

//---------------------------------------------------------------------------//
int
mpi_rank()
{
  int rank;
  MPI_Comm comm = MPI_Comm_f2c(mpi_comm());
  MPI_Comm_rank(comm, &rank);
  return rank;
}

//---------------------------------------------------------------------------//
int
mpi_size()
{
  int size;
  MPI_Comm comm = MPI_Comm_f2c(mpi_comm());
  MPI_Comm_size(comm, &size);
  return size;
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
#else // non-mpi case
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
void
check_comm()
{
  std::stringstream msg;
  msg<<"APComp internal error. Trying to access MPI comm in non-mpi lib.";
  msg<<"Are you using the right library (apcomp vs apcomp_mpi)?";
  throw Error(msg.str());
}

//---------------------------------------------------------------------------//
void
mpi_comm(int mpi_comm_id)
{
  std::stringstream msg;
  msg<<"APComp internal error. Trying to access MPI comm in non-mpi lib.";
  msg<<"Are you using the right library (apcomp vs apcomp_mpi)?";
  throw Error(msg.str());
}

//---------------------------------------------------------------------------//
int
mpi_comm()
{
  std::stringstream msg;
  msg<<"APComp internal error. Trying to access MPI comm in non-mpi lib.";
  msg<<"Are you using the right library (apcomp vs apcomp_mpi)?";
  throw Error(msg.str());
  return g_mpi_comm_id;
}

//---------------------------------------------------------------------------//
int
mpi_rank()
{
  return 0;
}

//---------------------------------------------------------------------------//
int
mpi_size()
{
  return 1;
}
//---------------------------------------------------------------------------//
#endif
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
bool
mpi_enabled()
{
#ifdef APCOMP_PARALLEL
  return true;
#else
  return false;
#endif
}

bool
openmp_enabled()
{
#ifdef APCOMP_OPENMP_ENABLED
  return true;
#else
  return false;
#endif
}

std::string about()
{
  std::string res;
  res = "APComp \n";
  if(mpi_enabled())
  {
    res += "mpi enabled\n";
  }
  else
  {
    res += "mpi disabled\n";
  }
  if(openmp_enabled())
  {
    res += "openmp enabled\n";
  }
  else
  {
    res += "openmp disabled\n";
  }
  return res;
}

} // namespace apcomp
