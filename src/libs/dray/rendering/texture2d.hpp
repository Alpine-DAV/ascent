// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TEXTURE2D_HPP
#define DRAY_TEXTURE2D_HPP

#include <dray/dray_config.h>
#include <dray/dray_exports.h>

#include <dray/array.hpp>
#include <dray/vec.hpp>
#include <dray/rendering/device_texture2d.hpp>

namespace dray
{

// types of textures we might want
// scalar (i.e, signed distance)
// rgb (this)
// rgba
class Texture2d
{
protected:
  int32 m_width;
  int32 m_height;
  Array<Vec<float32,3>> m_texture;
  int32 m_id;
public:
  Texture2d();
  Texture2d(const std::string png_file);
  DeviceTexture2D<Vec<float32,3>> device();
  int32 id();
  void id(const int32 i);
};


} // namespace dray
#endif
