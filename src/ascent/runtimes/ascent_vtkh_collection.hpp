//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef ASCENT_VTKH_COLLECTION_HPP
#define ASCENT_VTKH_COLLECTION_HPP
//-----------------------------------------------------------------------------
///
/// file: ascent_vtkh_collection.hpp
///
//-----------------------------------------------------------------------------

#include <ascent_exports.h>
#include <vtkh/DataSet.hpp>
#include <map>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{
//
// VTKH data collection is used to support mutliple topologies, also known as
// cell sets in vtkm, which is supported in Blueprint. The current use case is
// data sets that have structured grids and unstructured points. Blueprint
// enforces that fields be associated with a topology, and that there all
// fields have a unique name. Therefore its not possible that an 'energy'
// field to exist in two different topologies.
//
// From a vtkm point of view, each topology and associated fields are
// a distinct data set and can be treated as such within pipelines.
//
class ASCENT_API VTKHCollection
{
protected:
  std::map<std::string, vtkh::DataSet> m_datasets;
public:
  void add(vtkh::DataSet &dataset, const std::string topology_name);

  // returns true if the topology exists on any rank
  bool has_topology(const std::string name) const;

  // returns true if the field exists on any rank
  bool has_field(const std::string field_name) const;

  // returns the local summary
  std::string summary() const;

  // returns empty string if field not present on
  // any rank
  std::string field_topology(const std::string field_name);

  // returns an empty dataset if topology does not exist on
  // this rank
  vtkh::DataSet &dataset_by_topology(const std::string topology_name);

  vtkm::Bounds global_bounds() const;

  // returns the local topology names
  std::vector<std::string> topology_names() const;

  // returns the local domain ids
  std::vector<vtkm::Id> domain_ids() const;

  // returns the local number of topologies
  int number_of_topologies() const;

  int cycle() const;

  // returns a new collection without the specified topology
  // this is a shallow copy operation
  VTKHCollection* copy_without_topology(const std::string topology_name);

  // re-organize by 'domian_id / topology / data set'
  std::map<int, std::map<std::string,vtkm::cont::DataSet>> by_domain_id();

};

//-----------------------------------------------------------------------------
};
#endif
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
