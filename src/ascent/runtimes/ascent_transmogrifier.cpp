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

bool Transmogrifier::is_poly(const conduit::Node &doms)
{
  const int num_domains = doms.number_of_children();

  for (int i = 0; i < num_domains; i ++)
  {
    const conduit::Node &dom = doms.child(i);
    conduit::NodeConstIterator itr = dom["topologies"].children();
    while (itr.has_next())
    {
      const conduit::Node &topo = itr.next();
      if (topo.has_child("elements"))
      {
        if (topo["elements"].has_child("shape"))
        {
          if (topo["elements/shape"].as_string() == "polyhedral" || 
              topo["elements/shape"].as_string() == "polygonal")
          {
            return true;
          }
        }
      }
    }
  }

  return false;
}

// to_poly assumes that the node n is polyhedral
void Transmogrifier::to_poly(conduit::Node &doms, conduit::Node &to_vtkh)
{
  const int num_domains = doms.number_of_children();

  for (int i = 0; i < num_domains; i ++)
  {
    const conduit::Node &dom = doms.child(i);
    std::vector<std::string> poly_topos;
    conduit::Node &res = to_vtkh.append();

    // we know it must have a child "topologies" b/c otherwise it is not valid blueprint
    conduit::NodeConstIterator itr = dom["topologies"].children();
    while (itr.has_next())
    {
      const conduit::Node &topo = itr.next();
      if (topo.has_child("elements"))
      {
        if (topo["elements"].has_child("shape"))
        {
          if (topo["elements/shape"].as_string() == "polyhedral" || 
              topo["elements/shape"].as_string() == "polygonal")
          {
            poly_topos.push_back(topo.name());
          }
        }
      }
    }

    res.set_external(dom);

    std::vector<std::string> coordsets;
    for (int i = 0; i < poly_topos.size(); i ++)
    {
      conduit::Node s2dmap, d2smap, options;
      coordsets.push_back(dom["topologies/" + poly_topos[i] + "/coordset"].as_string());
      conduit::blueprint::mesh::topology::unstructured::generate_sides(
        dom["topologies/" + poly_topos[i]],
        res["topologies/" + poly_topos[i]],
        res["coordsets/" + coordsets[coordsets.size() - 1]],
        res["fields"],
        s2dmap,
        d2smap,
        options);
    }
  }
}

//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
