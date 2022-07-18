// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FONT_HPP
#define DRAY_FONT_HPP

#include <dray/types.hpp>
#include <dray/aabb.hpp>
#include <dray/array.hpp>
#include <dray/vec.hpp>

#include <conduit.hpp>
#include<vector>

namespace dray
{

class Font
{
protected:
  std::vector<float32> m_image;
  conduit::Node m_metadata;
  bool m_valid;
  float m_font_size;
  float m_native_font_size;

public:
  Font();
  Font(const std::string metrics,
       const unsigned char *png,
       size_t png_size);
  void font_size(const float size);
  float font_size() const;
  float native_font_size() const;

  AABB<2> font_boxs(const std::string text,
                    const Vec<float32,2> &pos,
                    std::vector<AABB<2>> &pixel_boxs,
                    std::vector<AABB<2>> &texture_boxs);

  Array<float32> texture();
  int32 texture_width() const;
  int32 texture_height() const;

  void load(const std::string metrics,
            const unsigned char *png,
            size_t png_size);
  bool valid() const;
};

} // namespace dray
#endif
