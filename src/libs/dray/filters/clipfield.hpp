// Copyright 2022 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#ifndef DRAY_CLIPFIELD_HPP
#define DRAY_CLIPFIELD_HPP

#include <dray/data_model/collection.hpp>
#include <string>

namespace dray
{

class ClipField
{
protected:
  Float m_clip_value;
  std::string   m_field_name;
  bool m_invert;
  bool m_exclude_clip_field;
public:
  ClipField();
  ~ClipField();

  void set_clip_value(Float value);
  void set_field(const std::string &field);
  void set_invert_clip(bool value);

  Float clip_value() const;
  const std::string &field() const;
  bool invert() const;
  void exclude_clip_field(bool);

  /**
   * @brief Execute on a single DataSet.
   */
  DataSet execute(DataSet domain);

  /**
   * @brief Execute on all DataSets in a Collection.
   * @param collection The collection.
   * @return A new collection with the clipped domains.
   *
   */
  Collection execute(Collection &collection);
};

};//namespace dray

#endif//DRAY_CLIPFIELD_HPP
