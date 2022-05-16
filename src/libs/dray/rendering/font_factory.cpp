// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <map>
#include <dray/rendering/font_factory.hpp>
#include <dray/rendering/fonts/open_sans.hpp>
#include <dray/error.hpp>

namespace dray
{

Font* FontFactory::font(const std::string &font_name)
{
  static std::map<std::string,Font> font_store;
  Font *res = nullptr;
  auto find = font_store.find(font_name);
  if(find != font_store.end())
  {
    return &find->second;
  }
  else
  {
    Font& new_font = font_store[font_name];
    new_font.load(opensans_metrics, opensans_png, opensans_png_len);
    res = &new_font;
  }

  return res;
}

}// namespace dray

