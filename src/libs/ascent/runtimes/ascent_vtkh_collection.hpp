//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
  std::map<std::string, vtkm::cont::PartitionedDataSet> p_datasets;
public:
  VTKHCollection();
  void add(vtkm::cont::PartitionedDataSet &dataset, const std::string topology_name);

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
  vtkm::cont::PartitionedDataSet &dataset_by_topology(const std::string topology_name);

  vtkm::Bounds global_bounds() const;

  // returns the local topology names
  std::vector<std::string> topology_names() const;

  // returns the local field names
  std::vector<std::string> field_names() const;

  // returns the local domain ids
  std::vector<vtkm::Id> domain_ids() const;

  // returns the local number of topologies
  int number_of_topologies() const;

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
