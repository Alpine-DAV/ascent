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

using namespace std;


//-----------------------------------------------------------------------------
TEST(apcomp_zbuffer, apcomp_zbuffer)
{
  std::string output_dir = prepare_output_dir();
  std::string file_name = "apcomp_zbuffer";
  std::string output_file = apcomp::join_file_path(output_dir,file_name);
  // TOD: Fix these, i don't think they delete
  remove_test_file(output_file);

  apcomp::Compositor compositor;
  auto mode = apcomp::Compositor::CompositeMode::Z_BUFFER_SURFACE_GL;
  compositor.SetCompositeMode(mode);

  const int width  = 1024;
  const int height = 1024;
  const int square_size = 300;
  const int num_images  = 4;
  const int y = 400;
  for(int i = 0; i < num_images; ++i)
  {

    float color[4];
    color[0] = 0.1f + float(i) * 0.1f;
    color[1] = 0.1f + float(i) * 0.1f;
    color[2] = 0.1f + float(i) * 0.1f;
    color[3] = 1.f;
    std::vector<float> pixels;
    std::vector<float> depths;
    gen_float32_image(pixels,
                      depths,
                      width,
                      height,
                      float(i) * 0.05f,
                      200 + 100*i,
                      y,
                      square_size,
                      color);

    compositor.AddImage(&pixels[0], &depths[0], width, height);
  }
  apcomp::Image image = compositor.Composite();
  image.Save(output_file);

EXPECT_TRUE(check_test_image(output_file, t_apcomp_baseline_dir()));

}

