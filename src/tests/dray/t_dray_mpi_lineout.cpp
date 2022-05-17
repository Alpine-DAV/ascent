// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include <dray/dray.hpp>
#include <mpi.h>

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>
#include <dray/queries/lineout.hpp>

#include <dray/math.hpp>

using namespace dray;

TEST (dray_mpi_lineout, dray_lineout)
{
  //
  // Set Up MPI
  //
  MPI_Comm comm = MPI_COMM_WORLD;
  ::dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "slice_scalars");
  remove_test_image (output_file);

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "taylor_green.cycle_001860.root";

  Collection collection = BlueprintReader::load (root_file);

  Lineout lineout;

  lineout.samples(10);
  lineout.add_var("density");
  // the data set bounds are [0,1] on each axis
  Vec<Float,3> start = {{0.01f,0.5f,0.5f}};
  Vec<Float,3> end = {{0.99f,0.5f,0.5f}};
  lineout.add_line(start, end);

  Lineout::Result res = lineout.execute(collection);

  if(::dray::dray::mpi_rank() == 0)
  for(int i = 0; i < res.m_values[0].size(); ++i)
  {
    std::cout<<"Value "<<i<<" "<<res.m_values[0].get_value(i)<<"\n";
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
