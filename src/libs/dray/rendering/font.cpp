// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/array.hpp>
#include <dray/error.hpp>
#include <dray/rendering/font.hpp>
#include <dray/utils/png_decoder.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/array_utils.hpp>

#include<vector>
#include<cmath>
#include<string.h>

namespace dray
{

Font::Font()
  : m_valid(false),
    m_font_size(16.f)
{
}

Font::Font(const std::string metrics,
          const unsigned char *png,
          size_t png_size)
  : m_valid(false),
    m_font_size(16.f)
{
  load(metrics, png, png_size);
}

bool Font::valid() const
{
  return m_valid;
}

void Font::font_size(const float size)
{
  m_font_size = size;
}

float32 Font::font_size() const
{
  return m_font_size;
}

float Font::native_font_size() const
{
  return m_native_font_size;
}

void Font::load(const std::string metrics,
                const unsigned char *png,
                size_t png_size)
{

  m_metadata.reset();
  m_valid = false;

  try
  {
    m_metadata.parse(metrics, "yaml");
    m_native_font_size = m_metadata["size"].to_float32();
    const int width = m_metadata["bitmap_width"].to_int32();
    const int height = m_metadata["bitmap_height"].to_int32();

    m_image.resize(width * height);

    PNGDecoder decoder;
    uint8 *buffer = nullptr;
    int b_width, b_height;
    decoder.decode_raw(buffer, b_width, b_height, png, png_size);
    if(b_width != width || b_height != height)
    {
      std::cout<<"Mismatched image dims\n";
    }


    for(int i = 0; i < width * height; ++i)
    {
      m_image[i] = float32(buffer[i*4+0]) / 255.f;
    }
    free(buffer);
    m_valid = true;
  }
  catch(const conduit::Error &e)
  {
    DRAY_ERROR("Failed to load font "<<e.what());
  }
  catch (...)
  {
    DRAY_ERROR("Unknown failure: load font");
  }
}

int32 Font::texture_width() const
{
  return m_metadata["bitmap_width"].to_int32();
}

int32 Font::texture_height() const
{
  return m_metadata["bitmap_height"].to_int32();
}


AABB<2> Font::font_boxs(const std::string text,
                        const Vec<float32,2> &pos,
                        std::vector<AABB<2>> &pixel_boxs,
                        std::vector<AABB<2>> &texture_boxs)
{

  if(!m_valid)
  {
    DRAY_ERROR("Font invalid\n");
  }
  Vec<float32,2> pen = pos;
  AABB<2> tot_aabb;
  std::string prev_char;
  for (auto it = text.begin(); it != text.end(); ++it)
  {
    std::string character = std::string(1,*it);
    if(!m_metadata.has_path("glyph_data/"+character))
    {
      // don't fail
      character = "x";
    }
    const conduit::Node &glyph = m_metadata["glyph_data/"+std::string(1,*it)];

    float kerning = 0.f;
    if(it != text.begin() && glyph.has_path("kernings/"+prev_char))
    {
      kerning = glyph["kernings/"+prev_char].to_float32();
    }
    pen[0] += kerning * m_font_size;

    float32 width     = glyph["bbox_width"].to_float32() * m_font_size;
    float32 height    = glyph["bbox_height"].to_float32() * m_font_size;
    float32 bearing_x = glyph["bearing_x"].to_float32() * m_font_size;
    float32 bearing_y = glyph["bearing_y"].to_float32() * m_font_size;
    float32 advance_x = glyph["advance_x"].to_float32() * m_font_size;
    float32 x = pen[0] + bearing_x;
    float32 y = pen[1] + bearing_y;
    float32 w = width;
    float32 h = height;
    pen[0] += advance_x;

    AABB<2> texture_box;
    texture_box.m_ranges[0].include(glyph["s0"].to_float32());
    texture_box.m_ranges[0].include(glyph["s1"].to_float32());
    texture_box.m_ranges[1].include(1.f - glyph["t0"].to_float32());
    texture_box.m_ranges[1].include(1.f - glyph["t1"].to_float32());
    texture_boxs.push_back(texture_box);

    AABB<2> pixel_box;
    pixel_box.m_ranges[0].include(x);
    pixel_box.m_ranges[0].include(x+w);
    pixel_box.m_ranges[1].include(y-h);
    pixel_box.m_ranges[1].include(y);
    pixel_boxs.push_back(pixel_box);
    tot_aabb.include(pixel_box);
    prev_char = *it;
  }
  return tot_aabb;
}

Array<float32> Font::texture()
{
  // Fonts are kepts around statically, but we can't
  // have static dray::Arrays because the umpire deallocators
  // are gone when the the program is destructed.
  // Decoding the png is more expensive, so we keep a
  // static vector of the decoded image instead
  Array<float32> texture;
  texture.resize(m_image.size());
  float32 *texture_ptr = texture.get_host_ptr();

  memcpy(texture_ptr, &m_image[0], sizeof(float32) * m_image.size());
  return texture;
}

} // namespace dray
