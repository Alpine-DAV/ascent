// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/dray.hpp>
#include <mpi.h>

TEST (dray_mpi_smoke, dray_about)
{
  //
  // Set Up MPI
  //
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));
  int par_rank = dray::dray::mpi_rank();
  int par_size = dray::dray::mpi_size();

  std::cout<<"Rank "
              << par_rank
              << " of "
              << par_size
              << " reporting\n";

  if(par_rank == 0)
  {
    dray::dray::about();
  }

}

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
