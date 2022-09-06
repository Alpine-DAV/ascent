// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/texture2d.hpp>
#include <dray/utils/png_decoder.hpp>
#include <dray/error.hpp>
#include <conduit.hpp>

namespace dray
{

Texture2d::Texture2d()
  : m_width(0),
    m_height(0),
    m_id(-1)
{

}

int32 Texture2d::id()
{
  return m_id;
}

void Texture2d::id(const int32 i)
{
  m_id = i;
}

Texture2d::Texture2d(const std::string png_file)
{
  bool exists = conduit::utils::is_file (png_file);
  if(!exists)
  {
    DRAY_ERROR("png file '"<<png_file<<"' does not exist");
  }

  int32 w, h;
  unsigned char *buff = nullptr;
  PNGDecoder decoder;
  decoder.decode(buff, w, h, png_file);

  m_width = w;
  m_height = h;
  const int32 size = w * h;
  m_texture.resize(size);
  Vec<float32,3> *t_ptr = m_texture.get_host_ptr();

  for(int32 i = 0; i < size; ++i)
  {
    const int32 offset = i * 4;
    Vec<float32,3> pixel;
    pixel[0] = float32(buff[offset + 0]) / 255.f;
    pixel[1] = float32(buff[offset + 1]) / 255.f;
    pixel[2] = float32(buff[offset + 2]) / 255.f;
    t_ptr[i] = pixel;
  }

}

DeviceTexture2D<Vec<float32,3>>
Texture2d::device()
{
  return DeviceTexture2D<Vec<float32,3>>(m_texture, m_width, m_height);
}

} // namespace dray
