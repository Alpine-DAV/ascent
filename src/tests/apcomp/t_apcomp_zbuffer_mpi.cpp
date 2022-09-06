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
#include <apcomp/compositor.hpp>

#include <iostream>
#include <mpi.h>

using namespace std;


//-----------------------------------------------------------------------------
TEST(apcomp_zbuffer_mpi, apcomp_zbuffer_mpi)
{
  std::string output_dir = prepare_output_dir();
  std::string file_name = "apcomp_zbuffer_mpi";
  std::string output_file = apcomp::join_file_path(output_dir,file_name);
  remove_test_file(output_file);

  int par_rank;
  int par_size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &par_rank);
  MPI_Comm_size(comm, &par_size);
  apcomp::mpi_comm(MPI_Comm_c2f(comm));

  apcomp::Compositor compositor;
  auto mode = apcomp::Compositor::CompositeMode::Z_BUFFER_SURFACE_GL;
  compositor.SetCompositeMode(mode);

  const int width  = 1024;
  const int height = 1024;
  const int square_size = 300;
  const int num_images  = 4;
  const int y = 400;

  float color[4];
  color[0] = 0.1f + float(par_rank) * 0.1f;
  color[1] = 0.1f + float(par_rank) * 0.1f;
  color[2] = 0.1f + float(par_rank) * 0.1f;
  color[3] = 1.f;
  std::vector<float> pixels;
  std::vector<float> depths;
  gen_float32_image(pixels,
                    depths,
                    width,
                    height,
                    float(par_rank) * 0.05f,
                    200 + 100 * par_rank,
                    y,
                    square_size,
                    color);

  compositor.AddImage(&pixels[0], &depths[0], width, height);

  apcomp::Image image = compositor.Composite();
  if(par_rank == 0)
  {
    image.Save(output_file);
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
