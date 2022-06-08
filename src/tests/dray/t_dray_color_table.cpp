// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "gtest/gtest.h"

#include "t_utils.hpp"
#include "t_config.hpp"

#include <dray/color_table.hpp>
#include <dray/utils/png_encoder.hpp>

using namespace dray;

void write_color_table (const std::string name, const std::string file_name)
{
  dray::ColorTable color_table (name);

  const int samples = 1024;
  Array<Vec<float32, 4>> color_map;
  color_table.sample (samples, color_map);


  const int width = 1024;
  const int height = 100;
  Array<Vec<float32, 4>> color_buffer;
  color_buffer.resize (width * height);

  const Vec<float32, 4> *color_ptr = color_map.get_host_ptr ();
  Vec<float32, 4> *buffer_ptr = color_buffer.get_host_ptr ();

  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      float32 t = float32 (x) / float32 (width);
      int32 color_idx = static_cast<int32> (t * (float32 (samples) - 1.f));
      int32 buffer_idx = y * width + x;
      buffer_ptr[buffer_idx] = color_ptr[color_idx];
    }
  }

  PNGEncoder encoder;
  const float32 *start = &(buffer_ptr[0][0]);
  encoder.encode (start, width, height);
  encoder.save (file_name + ".png");
}

TEST (dray_test, dray_color_table)
{
  std::string output_path = prepare_output_dir ();

  std::vector<std::string> color_tables;
  // builtin
  color_tables.push_back ("cool2warm");
  // additional
  color_tables.push_back ("3-wave-muted");

  for (int i = 0; i < color_tables.size (); ++i)
  {
    std::string output_file =
    conduit::utils::join_file_path (output_path, color_tables[i]);
    remove_test_image (output_file);
    write_color_table (color_tables[i], output_file);
    // check that we created an image
    EXPECT_TRUE (check_test_image (output_file,dray_baselines_dir()));
  }
}
