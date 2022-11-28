//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
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
