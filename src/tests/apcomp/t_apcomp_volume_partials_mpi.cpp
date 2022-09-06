//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_apcomp_zbuffer.cpp
///
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include "t_config.hpp"
#include "t_utils.hpp"
#include "t_apcomp_test_utils.h"

#include <apcomp/apcomp.hpp>
#include <apcomp/partial_compositor.hpp>

#include <iostream>
#include <mpi.h>

using namespace std;


//-----------------------------------------------------------------------------
TEST(apcomp_vpartial_mpi, apcomp_vpartial_mpi)
{
  std::string output_dir = prepare_output_dir();
  std::string file_name = "apcomp_volume_partial_mpi";
  std::string output_file = apcomp::join_file_path(output_dir,file_name);
  remove_test_file(output_file);

  int par_rank;
  int par_size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &par_rank);
  MPI_Comm_size(comm, &par_size);
  apcomp::mpi_comm(MPI_Comm_c2f(comm));
  if(par_size > 4)
  {
    EXPECT_TRUE(false);
  }

  apcomp::PartialCompositor<apcomp::VolumePartial<float>> compositor;

  const int width  = 1024;
  const int height = 1024;
  const int square_size = 300;
  const int num_images  = 4;
  const int y = 500;
  float colors[4][4] = { {1.f, 0.f, 0.f, 0.5f},
                         {0.f, 1.f, 0.f, 0.5f},
                         {0.f, 0.f, 1.f, 0.5f},
                         {0.f, 1.f, 1.f, 0.5f} } ;

  std::vector<std::vector<apcomp::VolumePartial<float>>> in_partials;
  in_partials.resize(1);

  std::vector<apcomp::VolumePartial<float>> partials;
  gen_float32_partials(in_partials[0],
                       width,
                       height,
                       float(par_rank) * 0.05f,
                       200 + 100*par_rank,
                       y - par_rank * 50,
                       square_size,
                       colors[par_rank]);


  std::vector<apcomp::VolumePartial<float>> output;
  compositor.composite(in_partials, output);

  if(par_rank == 0)
  {
    partials_to_png(output, width, height, output_file);
    EXPECT_TRUE(check_test_image(output_file, t_apcomp_baseline_dir()));
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
