// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"

#include "t_utils.hpp"

#include <dray/dray.hpp>
#include <dray/color_map.hpp>
#include <dray/rendering/font.hpp>
#include <dray/rendering/screen_text_annotator.hpp>
#include <dray/rendering/color_bar_annotator.hpp>

TEST (dray_smoke, dray_font)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "font_test");
  remove_test_image (output_file);

  dray::ScreenTextAnnotator annot;

  dray::Vec<dray::float32,2> pos({{9.f,1000.f}});

  for(float size = 100; size > 6.f; size = size * 0.9f)
  {
    pos[1] -= size;
    annot.add_text("Bananas", pos, size);
  }


  dray::Framebuffer fb;
  dray::Vec<float,4> color({{0.f, 0.f, 1.f, 1.f}});
  fb.foreground_color(color);

  fb.clear();
  annot.render(fb);

  dray::ColorTable ctable("cool2warm");
  ctable.add_alpha(0.0f, 0.0f);
  ctable.add_alpha(1.0f, 1.0f);
  dray::ColorMap cmap;
  cmap.color_table(ctable);

  dray::AABB<2> box;
  box.m_ranges[0].include(0);
  box.m_ranges[0].include(50);
  box.m_ranges[1].include(0);
  box.m_ranges[1].include(200);

  dray::Vec<dray::float32,2> box_size({{50,200}});
  dray::Vec<dray::float32,2> cpos({{900.f,800.f}});
  dray::ColorBarAnnotator cbar;
  cbar.render(fb, cmap.colors(), cpos, box_size);

  fb.composite_background();
  fb.save(output_file);
}
