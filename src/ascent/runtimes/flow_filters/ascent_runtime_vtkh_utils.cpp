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
/// file: ascent_runtime_vtkh_utils.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_vtkh_utils.hpp"
#include <ascent_runtime_utils.hpp>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

namespace detail
{


void field_error(const std::string field_name,
                 const std::string filter_name,
                 std::shared_ptr<VTKHCollection> collection,
                 bool error)
{
  std::string fpath = filter_to_path(filter_name);
  std::vector<std::string> names = collection->field_names();
  std::stringstream ss;
  ss<<" field names: ";
  for(int i = 0; i < names.size(); ++i)
  {
    ss<<"'"<<names[i]<<"'";
    if(i != names.size() - 1)
    {
      ss<<", ";
    }
  }
  if(error)
  {
    ASCENT_ERROR("("<<fpath<<") unknown field '"<<field_name<<"'"
                 <<ss.str());
  }
  else
  {
    ASCENT_INFO("("<<fpath<<") unknown field '"<<field_name<<"'"
                <<ss.str());
  }
}

std::string possible_topologies(std::shared_ptr<VTKHCollection> collection)
{
   std::stringstream ss;
   ss<<" topology names: ";
   std::vector<std::string> names = collection->topology_names();
   for(int i = 0; i < names.size(); ++i)
   {
     ss<<"'"<<names[i]<<"'";
     if(i != names.size() -1)
     {
       ss<<", ";
     }
   }
   return ss.str();
}

std::string resolve_topology(const conduit::Node &params,
                             const std::string filter_name,
                             std::shared_ptr<VTKHCollection> collection,
                             bool error)
{
  int num_topologies = collection->number_of_topologies();
  std::string topo_name;
  std::string fpath = filter_to_path(filter_name);
  if(num_topologies > 1)
  {
    if(!params.has_path("topology"))
    {
      std::string topo_names = detail::possible_topologies(collection);;
      if(error)
      {
        ASCENT_ERROR(fpath<<": data set has multiple topologies "
                     <<"and no topology is specified. "<<topo_names);
      }
      else
      {
        ASCENT_INFO(fpath<<": data set has multiple topologies "
                     <<"and no topology is specified. "<<topo_names);
        return topo_name;
      }
    }

    topo_name = params["topology"].as_string();
    if(!collection->has_topology(topo_name))
    {
      std::string topo_names = detail::possible_topologies(collection);;
      if(error)
      {
        ASCENT_ERROR(fpath<<": no topology named '"<<topo_name<<"'."
                     <<topo_names);
      }
      else
      {
        ASCENT_INFO(fpath<<": no topology named '"<<topo_name<<"'."
                   <<topo_names);
        return topo_name;
      }

    }
  }
  else
  {
    topo_name = collection->topology_names()[0];
  }

  return topo_name;
}

} // namespace detail
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
