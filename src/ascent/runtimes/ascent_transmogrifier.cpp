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
/// file: ascent_transmogrifier.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_transmogrifier.hpp"
#include "ascent_config.h"
#if defined(ASCENT_MFEM_ENABLED)
#include "ascent_mfem_data_adapter.hpp"
#endif
#include "ascent_logging.hpp"
#include <conduit_blueprint.hpp>
#include <algorithm>

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

int Transmogrifier::m_refinement_level = 3;

bool Transmogrifier::is_high_order(const conduit::Node &doms)
{
  // treat everything as a multi-domain data set
  const int num_domains = doms.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = doms.child(i);
    if(dom.has_path("fields"))
    {
      const conduit::Node &fields = dom["fields"];
      const int num_fields= fields.number_of_children();
      for(int t = 0; t < num_fields; ++t)
      {
        const conduit::Node &field = fields.child(t);
        if(field.has_path("basis")) return true;
      }

    }
  }

  return false;
}

conduit::Node* Transmogrifier::low_order(conduit::Node &dataset)
{
  if(!is_high_order(dataset))
  {
    ASCENT_ERROR("low_order requires high order data");
  }
#if defined(ASCENT_MFEM_ENABLED)
  MFEMDomains *domains = MFEMDataAdapter::BlueprintToMFEMDataSet(dataset);
  conduit::Node *lo_dset = new conduit::Node;
  MFEMDataAdapter::Linearize(domains, *lo_dset, m_refinement_level);
  delete domains;

  // add a second registry entry for the output so it can be zero copied.
  return lo_dset;
#else
  ASCENT_ERROR("Unable to convert high order mesh when MFEM is not enabled");
  return nullptr;
#endif
}

bool Transmogrifier::is_poly(const conduit::Node &n)
{
  if (! n.has_child("topologies"))
  {
    return false;
  }

  conduit::NodeConstIterator itr = n["topologies"].children();
  while (itr.has_next())
  {
    const conduit::Node &child = itr.next();
    if (child.has_child("elements/shape"))
    {
      if (child["elements/shape"].as_string() == "polyhedral" || 
          child["elements/shape"].as_string() == "polygonal")
      {
        return true;
      }
    }
  }

  return false;
}

// to_poly assumes that the node n is polyhedral
conduit::Node* Transmogrifier::to_poly(conduit::Node &n)
{
  std::vector<std::string> poly_topos;
  conduit::Node *res = new conduit::Node;
  // we know it must have a child "topologies" b/c of the check in is_poly()
  conduit::NodeConstIterator itr = n["topologies"].children();
  while (itr.has_next())
  {
    const conduit::Node &child = itr.next();
    if (child.has_child("elements/shape"))
    {
      if (child["elements/shape"].as_string() == "polyhedral" || 
          child["elements/shape"].as_string() == "polygonal")
      {
        poly_topos.push_back(child.name());
      }
    }
  }

  std::vector<std::string> coordsets;
  for (int i = 0; i < poly_topos.size(); i ++)
  {
    conduit::Node s2dmap, d2smap, options;
    coordsets.push_back(n["topologies/" + poly_topos[i] + "/coordset"].as_string());
    conduit::blueprint::mesh::topology::unstructured::generate_sides(
      n["topologies/" + poly_topos[i]],
      (*res)["topologies/" + poly_topos[i]],
      (*res)["coordsets/" + coordsets[coordsets.size() - 1]],
      (*res)["fields"],
      s2dmap,
      d2smap,
      options);
  }

  // we must copy all remaining pieces of the node as well

  // first we examine any children that are not coordsets, topologies, and fields
  itr = n.children();
  while (itr.has_next())
  {
    const conduit::Node &child = itr.next();
    std::string name = child.name();
    if (name != "coordsets" && name != "topologies" && name != "fields")
    {
      (*res)[name].set(child);
    }
  }

  // next we will look at any coordsets that may have been missed
  itr = n["coordsets"].children();
  while (itr.has_next())
  {
    const conduit::Node &child = itr.next();
    std::string name = child.name();
    // if this coordset is one of the ones we did not copy
    if (std::find(coordsets.begin(), coordsets.end(), name) == coordsets.end())
    {
      (*res)["coordsets/" + name].set(child);
    }
  }

  // next we will do the same for topologies
  itr = n["topologies"].children();
  while (itr.has_next())
  {
    const conduit::Node &child = itr.next();
    std::string name = child.name();
    // if the topology is *not* polyhedral
    if (child.has_child("elements/shape"))
    {
      if (!(child["elements/shape"].as_string() == "polyhedral" || 
                child["elements/shape"].as_string() == "polygonal"))
      {
        (*res)["topologies/" + name].set(child);
      }
    }
  }

  // and finally we will copy over any fields that were not copied
  // first we will figure out which ones were copied/mapped
  std::vector<std::string> copied_field_names;
  itr = (*res)["fields"].children();
  while (itr.has_next())
  {
    const conduit::Node &child = itr.next();
    copied_field_names.push_back(child.name());
  }
  // then we will copy over any that were not copied in generate_sides
  itr = n["fields"].children();
  while (itr.has_next())
  {
    const conduit::Node &child = itr.next();
    std::string name = child.name();
    // if the name from "n" is not in "res"
    if (std::find(copied_field_names.begin(), copied_field_names.end(), name) == copied_field_names.end())
    {
      (*res)["fields/" + name].set(child);
    }
  }

  return res;
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
