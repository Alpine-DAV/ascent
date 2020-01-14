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

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

void VTKHCollection::add(vtkh::DataSet &dataset, const std::string topology_name)
{
  if(has_topology(topology_name))
  {
    ASCENT_ERROR("VTKH collection already had topology '"<<topology_name<<"'");
  }
  m_datasets[topology_name] = dataset;
}

bool VTKHCollection::has_topology(const std::string name) const
{
  return m_datasets.count(name) != 0;
}

std::string VTKHCollection::field_topology(const std::string field_name)
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

  return topo_name;
}

int
VTKHCollection::cycle() const
{
  int cycle = 0;
  for(auto it = m_datasets.begin(); it != m_datasets.end(); ++it)
  {
    it->second.GetCycle();
    break;
  }
  return cycle;
}

vtkh::DataSet
VTKHCollection::dataset_by_topology(const std::string topology_name)
{
  if(!has_topology(topology_name))
  {
    ASCENT_ERROR("VTKH collection has no topology '"<<topology_name<<"'");
  }

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

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- end header ifdef guard
//-----------------------------------------------------------------------------


