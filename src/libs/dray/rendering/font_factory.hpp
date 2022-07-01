// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FONT_FACTORY_HPP
#define DRAY_FONT_FACTORY_HPP

#include <string>
#include <dray/rendering/font.hpp>

namespace dray
{

class FontFactory
{
public:
  static Font* font(const std::string &font_name);
};

} // namespace dray

#endif
