//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_mpi_utils.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_mpi_utils.hpp"
#include <flow.hpp>
#ifdef ASCENT_MPI_ENABLED
#include <conduit_relay_mpi.hpp>
#endif

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{
bool global_agreement(bool vote)
{
  bool agreement = vote;
#ifdef ASCENT_MPI_ENABLED
  int local_boolean = vote ? 1 : 0;
  int global_boolean;

  int comm_id = flow::Workspace::default_mpi_comm();
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean != mpi_size())
  {
    agreement = false;
  }
#endif
  return agreement;
}

bool global_someone_agrees(bool vote)
{
  bool agreement = vote;
#ifdef ASCENT_MPI_ENABLED
  int local_boolean = vote ? 1 : 0;
  int global_boolean;

  int comm_id = flow::Workspace::default_mpi_comm();
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean > 0)
  {
    agreement = true;
  }
  else
  {
    agreement = false;
  }
#endif
  return agreement;
}

int mpi_size()
{
  int comm_size = 1;
#ifdef ASCENT_MPI_ENABLED
  int comm_id = flow::Workspace::default_mpi_comm();
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
  MPI_Comm_size(mpi_comm, &comm_size);
#endif
  return comm_size;
}

int mpi_rank()
{
  int rank = 0;
#ifdef ASCENT_MPI_ENABLED
  int comm_id = flow::Workspace::default_mpi_comm();
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
  MPI_Comm_rank(mpi_comm, &rank);
#endif
  return rank;
}

void gather_strings(std::set<std::string> &string_set)
{
#ifdef ASCENT_MPI_ENABLED
  int comm_id = flow::Workspace::default_mpi_comm();
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);

  conduit::Node n_strings;
  for(auto &name : string_set)
  {
    n_strings.append() = name;
  }
  conduit::Node res;
  // this is going to give us a list (one item for each rank) of
  // lists (of string for each rank).
  conduit::relay::mpi::all_gather_using_schema(n_strings, res, mpi_comm);
  int ranks = res.number_of_children();
  for(int r = 0; r < ranks; ++r)
  {
    const conduit::Node &n_rank = res.child(r);
    const int num_children = n_rank.number_of_children();
    for(int i = 0; i < num_children; ++i)
    {
      string_set.insert(n_rank.child(i).as_string());
    }
  }
#endif
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------



