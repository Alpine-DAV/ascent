// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_CONTOUR_HPP
#define DRAY_CONTOUR_HPP

#include <dray/rendering/traceable.hpp>

namespace dray
{

class Contour : public Traceable
{
protected:
  std::string m_iso_field_name;
  float32 m_iso_value;
public:
  Contour() = delete;
  Contour(Collection &collection);
  virtual ~Contour();

  virtual Array<RayHit> nearest_hit(Array<Ray> &rays) override;

  void iso_field(const std::string field_name);
  void iso_value(const float32 iso_value);

};

};//namespace dray

#endif//DRAY_CONTOUR_HPP
