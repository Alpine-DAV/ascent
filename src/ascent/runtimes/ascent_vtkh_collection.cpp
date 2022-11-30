//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: ascent_vtkh_collection.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_vtkh_collection.hpp"
#include "ascent_mpi_utils.hpp"
#include "ascent_logging.hpp"

#if defined(ASCENT_MPI_ENABLED)
#include <mpi.h>
#include <conduit_relay_mpi.hpp>
#endif
//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{
namespace detail
{
//
// returns true if all ranks say true
//
bool global_has(bool local)
{
  bool has = local;
#if defined(ASCENT_MPI_ENABLED)
  int local_boolean = local ? 1 : 0;
  int global_boolean;
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean == 0)
  {
    has = false;
  }
  else
  {
    has = true;
  }
#endif
  return has;
}

int global_max(int local)
{
   int global_count = local;
#if defined(ASCENT_MPI_ENABLED)

  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Allreduce((void *)(&local),
                (void *)(&global_count),
                1,
                MPI_INT,
                MPI_MAX,
                mpi_comm);

#endif
  return global_count;
}

} // namespace detail

void VTKHCollection::add(vtkh::DataSet &dataset, const std::string topology_name)
{
  bool has_topo = p_datasets.count(topology_name) != 0;
  if(has_topo)
  {
    ASCENT_ERROR("VTKH collection already had topology '"<<topology_name<<"'");
  }
  p_datasets[topology_name] = dataset;
}

bool VTKHCollection::has_topology(const std::string name) const
{
  bool has_topo = p_datasets.count(name) != 0;

  return detail::global_has(has_topo);
}

std::string VTKHCollection::field_topology(const std::string field_name) {
  std::string topo_name = "";

  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    // Should we really have to ask an MPI questions? its safer
    if(it->second.HasField(field_name))
    {
      topo_name = it->first;
      break;
    }
  }
#if defined(ASCENT_MPI_ENABLED)
  // if the topology does not exist on this rank,
  // but exists somewhere, we need to figure out what
  // that name is for all ranks so all ranks
  // can run the same code at the same time, avoiding deadlock
  int rank;
  MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());
  MPI_Comm_rank(mpi_comm, &rank);

  struct MaxLoc
  {
    double size;
    int rank;
  };

  // there is no MPI_INT_INT so shove the "small" size into double
  MaxLoc maxloc = {(double)topo_name.length(), rank};
  MaxLoc maxloc_res;
  MPI_Allreduce( &maxloc, &maxloc_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi_comm);

  conduit::Node msg;
  msg["topo"] = topo_name;
  conduit::relay::mpi::broadcast_using_schema(msg,maxloc_res.rank,mpi_comm);

  if(!msg["topo"].dtype().is_string())
  {
    ASCENT_ERROR("failed to broadcast topo name");
  }
  topo_name = msg["topo"].as_string();
#endif
  return topo_name;
}

bool VTKHCollection::has_field(const std::string field_name) const
{
  bool has = false;
  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    // Should we really have to ask an MPI questions? its safer
    if(it->second.HasField(field_name))
    {
      has = true;
      break;
    }
  }


  return detail::global_has(has);
}

vtkm::Bounds VTKHCollection::global_bounds() const
{
  // ranks may have different numbers of local vtk-h datasets
  // depending on the toplogies at play
  // can't use vtk-h to get global bounds b/c could create
  // unmatched collectives.

  // to get the global bounds, we include all local bounds
  // then do a mpi reduce here
  vtkm::Bounds bounds;
  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    bounds.Include(GetBounds(it->second));
  }

#if defined(ASCENT_MPI_ENABLED)
    MPI_Comm mpi_comm = MPI_Comm_f2c(vtkh::GetMPICommHandle());

    vtkm::Float64 loc_mins[3];
    //x,y,z
    loc_mins[0] = bounds.X.Min;
    loc_mins[1] = bounds.Y.Min;
    loc_mins[2] = bounds.Z.Min;

    vtkm::Float64 loc_maxs[3];
    //x,y,z
    loc_maxs[0] = bounds.X.Max;
    loc_maxs[1] = bounds.Y.Max;
    loc_maxs[2] = bounds.Z.Max;

    vtkm::Float64 global_mins[3];
    //x,y,z
    global_mins[0] = 0.0;
    global_mins[1] = 0.0;
    global_mins[2] = 0.0;

    vtkm::Float64 global_maxs[3];
    //x,y,z
    global_maxs[0] = 0.0;
    global_maxs[1] = 0.0;
    global_maxs[2] = 0.0;

    MPI_Allreduce((void *)(&loc_mins),
                  (void *)(&global_mins),
                  3,
                  MPI_DOUBLE,
                  MPI_MIN,
                  mpi_comm);

    MPI_Allreduce((void *)(&loc_maxs),
                  (void *)(&global_maxs),
                  3,
                  MPI_DOUBLE,
                  MPI_MAX,
                  mpi_comm);

    bounds.X.Min = global_mins[0];
    bounds.X.Max = global_maxs[0];

    bounds.Y.Min = global_mins[1];
    bounds.Y.Max = global_maxs[1];

    bounds.Z.Min = global_mins[2];
    bounds.Z.Max = global_maxs[2];
  #endif

  return bounds;
}
//TODO:unclear how to count domains across multiple partitions
//do you ever want multiple partitions or just keep adding to the same partition?
std::vector<vtkm::Id> VTKHCollection::domain_ids() const
{
  std::vector<vtkm::Id> all_ids;
  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    vtkm::Id num_domains = it->second.GetNumberOfPartitions();
    for(int i = 0; i < num_domains; ++i)
    {
      all_ids.push_back(i);
    }
  }
  return all_ids;
}

vtkh::DataSet&
VTKHCollection::dataset_by_topology(const std::string topology_name)
{
  // this will return a empty dataset if this rank
  // does not actually have this topo, but it exists
  // globally
  return p_datasets[topology_name];
}

std::vector<std::string> VTKHCollection::topology_names() const
{
  std::set<std::string> names;
  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    names.insert(it->first);
  }
  gather_strings(names);
  std::vector<std::string> res(names.size());
  std::copy(names.begin(), names.end(), res.begin());
  return res;
}

std::vector<std::string> VTKHCollection::field_names() const
{
  // just grab the first domain of every topo and repo
  // the known fields
  std::set<std::string> names;
  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    vtkm::cont::PartitionedDataSet domains = it->second;
    if(domains.GetNumberOfPartitions() > 0)
    {
      vtkm::cont::DataSet dom = domains.GetPartition(0);
      for(int i = 0; i < dom.GetNumberOfFields(); ++i)
      {
        names.insert(dom.GetField(i).GetName());
      }
    }
  }

  gather_strings(names);
  std::vector<std::string> res(names.size());
  std::copy(names.begin(), names.end(), res.begin());
  return res;
}

std::map<int, std::map<std::string,vtkm::cont::DataSet>>
VTKHCollection::by_domain_id()
{
  std::map<int, std::map<std::string, vtkm::cont::DataSet>> res;

  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    const std::string topo_name = it->first;
    vtkm::cont::PartitionedDataSet &vtkm_pdataset = it->second;

    vtkm::Id num_domain = vtkm_pdataset.GetNumberOfPartitions();
    for(int i = 0; i < num_domains; ++i)
    {
      const int domain_id = i;
      res[domain_id][topo_name] = vtkm_pdataset.GetPartition(i);
    }
  }

  return res;
}

int VTKHCollection::number_of_topologies() const
{
  // this is not perfect. For example, we could
  // random topology names on different ranks,
  // but this is 99% of our possible use cases
  return detail::global_max(p_datasets.size());
}

VTKHCollection* VTKHCollection::copy_without_topology(const std::string topology_name)
{
  if(!has_topology(topology_name))
  {
    ASCENT_ERROR("Copy without topology with '"<<topology_name<<"' failed."
                 << " Topology does not exist");
  }

  VTKHCollection *copy = new VTKHCollection(*this);
  copy->p_datasets.erase(topology_name);

  return copy;
}

std::string VTKHCollection::summary() const
{
  std::stringstream msg;

  msg<<"vtkh colletion:\n";
  for(auto it = p_datasets.begin(); it != p_datasets.end(); ++it)
  {
    const std::string topo_name = it->first;
    msg<<"  Topology '"<<topo_name<<"': \n";
    const vtkm::cont::PartitionedDataSet &vtkm_pdataset = it->second;
    vtkm_pdataset.PrintSummary(msg);

  }
  return msg.str();
}

VTKHCollection::VTKHCollection()
{

}
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


