// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_COLLECTION_HPP
#define DRAY_COLLECTION_HPP

#include <dray/data_model/data_set.hpp>
#include <dray/types.hpp>
#include <map>

namespace dray
{

class Collection
{
protected:
  std::vector<DataSet> m_domains;
  AABB<3> m_bounds;
public:

  Collection();

  void add_domain(const DataSet &domain);
  DataSet domain(int32 index);

  struct DomainRange
  {
    Collection &m_collxn;
    typename std::vector<DataSet>::iterator begin() { return m_collxn.m_domains.begin(); }
    typename std::vector<DataSet>::iterator end() { return m_collxn.m_domains.end(); }
  };

  DomainRange domains() { return DomainRange{*this}; }

  Range range(const std::string field_name);
  Range local_range(const std::string field_name);

  bool has_field(const std::string field_name);
  bool local_has_field(const std::string field_name);

  AABB<3> bounds();
  AABB<3> local_bounds();

  int32 topo_dims();
  // total number of domains on all ranks
  int32 size();
  // number of domains on this rank
  int32 local_size();

  // get a string of all known fields
  std::string field_list();

};

} //namespace dray

#endif
