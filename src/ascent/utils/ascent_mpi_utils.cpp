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
/// file: ascent_mpi_utils.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_mpi_utils.hpp"
#include <flow.hpp>

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

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
