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

using namespace std;


//-----------------------------------------------------------------------------
TEST(apcomp_partials, apcomp_volume_partial)
{
  std::string output_dir = prepare_output_dir();
  std::string file_name = "apcomp_volume_partial";
  std::string output_file = conduit::utils::join_file_path(output_dir,file_name);
  remove_test_file(output_file);

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
  in_partials.resize(num_images);

  for(int i = 0; i < num_images; ++i)
  {
    std::vector<apcomp::VolumePartial<float>> partials;
    gen_float32_partials(in_partials[i],
                         width,
                         height,
                         float(i) * 0.05f,
                         200 + 100*i,
                         y - i * 50,
                         square_size,
                         colors[i]);

  }
  std::vector<apcomp::VolumePartial<float>> output;
  compositor.composite(in_partials, output);

  partials_to_png(output, width, height, output_file);

  EXPECT_TRUE(check_test_image(output_file, t_apcomp_baseline_dir()));
}

