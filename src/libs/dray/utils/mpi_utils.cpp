// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include <dray/utils/mpi_utils.hpp>
#include <dray/dray.hpp>

#ifdef DRAY_MPI_ENABLED
#include <conduit_relay_mpi.hpp>
#include <mpi.h>
#endif

namespace dray
{
bool global_agreement(bool vote)
{
  bool agreement = vote;
#ifdef DRAY_MPI_ENABLED
  int local_boolean = vote ? 1 : 0;
  int global_boolean;

  int comm_id = dray::mpi_comm();
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean != dray::mpi_size())
  {
    agreement = false;
  }
#endif
  return agreement;
}

bool global_someone_agrees(bool vote)
{
  bool agreement = vote;
#ifdef DRAY_MPI_ENABLED
  int local_boolean = vote ? 1 : 0;
  int global_boolean;

  int comm_id = dray::mpi_comm();
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


void gather_strings(std::set<std::string> &string_set)
{
#ifdef DRAY_MPI_ENABLED
  int comm_id = dray::mpi_comm();
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm_id);

  conduit::Node n_strings;
  for(auto &name : string_set)
  {
    n_strings[name] = 0;
  }
  conduit::Node res;
  conduit::relay::mpi::all_gather_using_schema(n_strings, res, mpi_comm);
  int num_children = res.number_of_children();
  for(int i = 0; i < num_children; ++i)
  {
    std::vector<std::string> res_names = res.child(i).child_names();
    for(auto &str : res_names)
    {
      string_set.insert(str);
    }
  }
#endif
}

}; // namespace dray



