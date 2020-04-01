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


//-----------------------------------------------------------------------------
///
/// file: ascent_vtkh_collection.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_vtkh_collection.hpp"
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
  bool has_topo = m_datasets.count(topology_name) != 0;
  if(has_topo)
  {
    ASCENT_ERROR("VTKH collection already had topology '"<<topology_name<<"'");
  }
  m_datasets[topology_name] = dataset;
}

bool VTKHCollection::has_topology(const std::string name) const
{
  bool has_topo = m_datasets.count(name) != 0;

  return detail::global_has(has_topo);
}

std::string VTKHCollection::field_topology(const std::string field_name) {
  std::string topo_name = "";

  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    // Should we really have to ask an MPI questions? its safer
    if(it->second.FieldExists(field_name))
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
  std::string topo_name = "";
  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    // Should we really have to ask an MPI questions? its safer
    if(it->second.GlobalFieldExists(field_name))
    {
      topo_name = it->first;
      break;
    }
  }

  return topo_name != "";
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
  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    bounds.Include(it->second.GetBounds());
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

std::vector<vtkm::Id> VTKHCollection::domain_ids() const
{
  std::vector<vtkm::Id> all_ids;
  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    std::vector<vtkm::Id> domain_ids = it->second.GetDomainIds();
    for(int i = 0; i < domain_ids.size(); ++i)
    {
      all_ids.push_back(domain_ids[i]);
    }
  }
  return all_ids;
}

int
VTKHCollection::cycle() const
{
  int cycle = 0;
  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    cycle = it->second.GetCycle();
    break;
  }
  return cycle;
}

vtkh::DataSet&
VTKHCollection::dataset_by_topology(const std::string topology_name)
{
  // this will return a empty dataset if this rank
  // does not actually have this topo, but it exists
  // globally
  return m_datasets[topology_name];
}

std::vector<std::string> VTKHCollection::topology_names() const
{
  std::vector<std::string> names;
  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
   names.push_back(it->first);
  }
  return names;
}

std::map<int, std::map<std::string,vtkm::cont::DataSet>>
VTKHCollection::by_domain_id()
{
  std::map<int, std::map<std::string, vtkm::cont::DataSet>> res;

  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    const std::string topo_name = it->first;
    vtkh::DataSet &vtkh_dataset = it->second;

    std::vector<vtkm::Id> domain_ids = vtkh_dataset.GetDomainIds();
    for(int i = 0; i < domain_ids.size(); ++i)
    {
      const int domain_id = domain_ids[i];
      res[domain_id][topo_name] = vtkh_dataset.GetDomain(domain_id);
    }
  }

  return res;
}

int VTKHCollection::number_of_topologies() const
{
  // this is not perfect. For example, we could
  // random topology names on different ranks,
  // but this is 99% of our possible use cases
  return detail::global_max(m_datasets.size());
}

VTKHCollection* VTKHCollection::copy_without_topology(const std::string topology_name)
{
  if(!has_topology(topology_name))
  {
    ASCENT_ERROR("Copy without topology with '"<<topology_name<<"' failed."
                 << " Topology does not exist");
  }

  VTKHCollection *copy = new VTKHCollection(*this);
  copy->m_datasets.erase(topology_name);

  return copy;
}

std::string VTKHCollection::summary() const
{
  std::stringstream msg;

  msg<<"vtkh colletion:\n";
  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    const std::string topo_name = it->first;
    msg<<"  Topology '"<<topo_name<<"': \n";
    const vtkh::DataSet &vtkh_dataset = it->second;
    vtkh_dataset.PrintSummary(msg);

  }
  return msg.str();
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


