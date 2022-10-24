// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ISOSURFACING_HPP
#define DRAY_ISOSURFACING_HPP

#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>

#include <utility>

namespace dray
{


class ExtractIsosurface
{
protected:
  std::string m_iso_field_name;
  Float m_iso_value;
  bool all_linear(Collection &collxn);
  std::pair<DataSet, DataSet> execute(DataSet &data_set);
public:
  std::pair<Collection, Collection> execute(Collection &collxn);

  void iso_field(const std::string field_name);
  std::string iso_field() const;

  void iso_value(const float32 iso_value);
  Float iso_value() const;
};

};//namespace dray


#endif
