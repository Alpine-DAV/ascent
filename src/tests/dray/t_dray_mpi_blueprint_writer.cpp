// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/dray.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <mpi.h>

TEST (dray_mpi_blueprint_writer, dray_write)
{
  //
  // Set Up MPI
  //
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));

  std::string root_file = std::string (ASCENT_T_DATA_DIR) + "laghos_tg.cycle_000350.root";
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "laghos_subset");

  conduit::Node dataset;
  dray::BlueprintReader::load_blueprint(root_file, dataset);

  conduit::Node subset;
  if(dataset.number_of_children() > 1)
  {
    conduit::Node &dom = subset.append();
    dom.set_external(dataset.child(0));
  }
  dray::BlueprintReader::save_blueprint(output_file, subset);

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
