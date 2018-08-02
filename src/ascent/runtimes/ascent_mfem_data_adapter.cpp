//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
/// file: ascent_mfem_data_adapter.cpp
///
//-----------------------------------------------------------------------------
#include "ascent_mfem_data_adapter.hpp"

#include <ascent_logging.hpp>

// standard lib includes
#include <iostream>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include <sstream>

// third party includes
#include <conduit_blueprint.hpp>
// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#include <mfem.hpp>

using namespace std;
using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

MFEMDataSet::MFEMDataSet()
{

}

MFEMDataSet::~MFEMDataSet()
{
  delete m_mesh;

  for(auto it = m_fields.begin(); it != m_fields.end(); ++it)
  {
    delete it->second;
  }
}

MFEMDataSet::MFEMDataSet(mfem::Mesh *mesh)
 : m_mesh(mesh)
{

}

void 
MFEMDataSet::set_mesh(mfem::Mesh *mesh)
{
  m_mesh = mesh;
}

mfem::Mesh*
MFEMDataSet::get_mesh()
{
  return m_mesh;
}

void 
MFEMDataSet::add_field(mfem::GridFunction *field, const std::string &name)
{
  m_fields[name] = field;
}

MFEMDataSet::FieldMap
MFEMDataSet::get_field_map()
{
  return m_fields;
}

bool 
MFEMDataSet::has_field(const std::string &field_name)
{
  auto it = m_fields.find(field_name);
  return it != m_fields.end();
}

mfem::GridFunction*
MFEMDataSet::get_field(const std::string &field_name)
{
  if(!has_field(field_name))
  {
    std::string msg = "MFEMDataSet: no field named : " + field_name;
    ASCENT_ERROR(msg);
  }
  
  return m_fields[field_name];
}

int 
MFEMDataSet::num_fields()
{
  return m_fields.size();
}

//-----------------------------------------------------------------------------
// MFEMDataAdapter public methods
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
MFEMDomains*
MFEMDataAdapter::BlueprintToMFEMDataSet(const Node &node,
                                        const std::string &topo_name)
{       
 
    // treat everything as a multi-domain data set 

    MFEMDomains *res = new MFEMDomains;


    // get the number of domains and check for id consistency
    const int num_domains = node.number_of_children();

    for(int i = 0; i < num_domains; ++i)
    {
      MFEMDataSet *dset = new MFEMDataSet();
      const conduit::Node &dom = node.child(i);      
      // this should exist
      int domain_id = dom["state/domain_id"].to_int();
      // insert domain conversion

      bool zero_copy = false;
      mfem::Mesh *mesh = nullptr;
      mesh = mfem::ConduitDataCollection::BlueprintMeshToMesh(dom, topo_name, zero_copy);
      dset->set_mesh(mesh);

      //dom.print();
      std::cout<<"MESH "<<mesh->GetNE()<<"\n";

      if(node.has_path("state/cycle"))
      {
        int cycle = node["state/cycle"].to_int32();
      }
  
      std::string t_name = topo_name;
      // no topology name provied, use the first
      if(t_name == "")
      {
        std::vector<std::string> tnames = dom["topologies"].child_names(); 
        if(tnames.size() > 0)
        {
          t_name = tnames[0];
        }
      }

      if(t_name == "")
      {
        ASCENT_ERROR("Unable to determine topology");
      }

      std::string nodes_gf_name = "";
      const Node &n_topo = dom["topologies/" + t_name];
      if (n_topo.has_child("grid_function"))
      {
        nodes_gf_name = n_topo["grid_function"].as_string();
      }
      if(dom.has_path("fields"))
      {
        const int num_fields = dom["fields"].number_of_children();
        std::vector<std::string> fnames = dom["fields"].child_names(); 
        for(int f = 0; f < num_fields; ++f)
        {
          const conduit::Node &field = dom["fields"].child(f);
           
          // skip any field that has a unsupported basis type 
          //      (we only supprt H1 (continuos) and L2 (discon)
          bool unsupported = false;
          if(field.has_child("basis"))
          {
            std::string basis = field["basis"].as_string();
            if(basis.find("H1_") == std::string::npos &&
               basis.find("L2_") == std::string::npos)
            {
              unsupported = true;
            }
          }
          // skip mesh nodes gf since they are already processed
          // skip attribute fields, they aren't grid functions
          if ( fnames[f] != nodes_gf_name && 
              fnames[f].find("_attribute") == std::string::npos &&
              !unsupported) 
          {
            mfem::GridFunction *gf = mfem::ConduitDataCollection::BlueprintFieldToGridFunction(mesh, 
                                                                                               field, 
                                                                                               zero_copy);
            dset->add_field(gf, fnames[f]); 
          }
        }
      }

      res->m_data_sets.push_back(dset);
      res->m_domain_ids.push_back(domain_id);
    
    }    
    return res;
}

// although is should probably never be the case that on domain
// is high order and the others are not, this method will 
// return true if at least one domian is higher order
bool 
MFEMDataAdapter::IsHighOrder(const conduit::Node &n)
{
  
  // treat everything as a multi-domain data set 
  const int num_domains = n.number_of_children();
  for(int i = 0; i < num_domains; ++i)
  {
    const conduit::Node &dom = n.child(i);      
    if(dom.has_path("fields"))
    {
      const conduit::Node &fields = dom["fields"];
      const int num_fields= fields.number_of_children();
      for(int t = 0; t < num_fields; ++t)
      {
        const conduit::Node &field = fields.child(i);      
        if(field.has_path("basis")) return true;
      }
      
    }
  }

  return false;
}
//                                          VDim
// +------------+--------------------+------------------+
// | Space Type | FE Collection Type | Vector Dimension |
// +------------+--------------------+------------------+
// | SC         | H1_FESpace         | 1                | scalar continous (nodes)
// +------------+--------------------+------------------+
// | SD         | L1                 | 1                | scalar discontinous (zonal sort of)
// +------------+--------------------+------------------+
// | VC         | H1                 | D                | vector continious
// +------------+--------------------+------------------+
// | TC         | SCColl             | (D*(D+1))/2      | tensor continous
// +------------+--------------------+------------------+
// | TD         | SDColl             | (D*(D+1))/2      | tensor discontinous
// +------------+--------------------+------------------+
// | RT         | RTColl             | 1                |
// +------------+--------------------+------------------+
// | ND         | NDColl             | 1                |
// +------------+--------------------+------------------+
void
MFEMDataAdapter::Linearize(MFEMDomains *ho_domains, conduit::Node &output, const int refinement)
{
  const int n_doms = ho_domains->m_data_sets.size();

  output.reset();
  for(int i = 0; i < n_doms; ++i)
  {

    conduit::Node &n_dset = output.append(); 
    n_dset["state/domain_id"] = int(ho_domains->m_domain_ids[i]);

    // get the high order data
    mfem::Mesh *ho_mesh = ho_domains->m_data_sets[i]->get_mesh(); 
    const mfem::FiniteElementSpace *ho_fes_space = ho_mesh->GetNodalFESpace();
    const mfem::FiniteElementCollection *ho_fes_col = ho_fes_space->FEColl();
    // refine the mesh and convert to blueprint
    mfem::Mesh *lo_mesh = new mfem::Mesh(ho_mesh, refinement, mfem::BasisType::GaussLobatto); 
    mfem::ConduitDataCollection::MeshToBlueprintMesh (lo_mesh, n_dset);
    //n_dset.print();
    //std::ofstream lovtk("low.vtk"); 
    //lo_mesh->PrintVTK(lovtk);
    
    int conn_size = n_dset["topologies/main/elements/connectivity"].dtype().number_of_elements();
    //int dims = ho_mesh->Dimension();

    conduit::Node &n_fields = n_dset["fields"];
    auto field_map = ho_domains->m_data_sets[i]->get_field_map();

    for(auto it = field_map.begin(); it != field_map.end(); ++it)
    {
      mfem::GridFunction *ho_gf = it->second;
      mfem::FiniteElementSpace *ho_fes = ho_gf->FESpace();
      if(ho_fes == nullptr) 
      {
        ASCENT_ERROR("Linearize: high order gf finite element space is null") 
      }
      // create the low order grid function
      mfem::FiniteElementCollection *lo_col = new mfem::LinearFECollection;
      mfem::FiniteElementSpace *lo_fes = new mfem::FiniteElementSpace(lo_mesh, lo_col, ho_fes->GetVDim());
      mfem::GridFunction *lo_gf = new mfem::GridFunction(lo_fes);
      // transform the higher order function to a low order function somehow
      mfem::OperatorHandle hi_to_lo;
      lo_fes->GetTransferOperator(*ho_fes, hi_to_lo);
      hi_to_lo.Ptr()->Mult(*ho_gf, *lo_gf);
      // extract field
      conduit::Node &n_field = n_fields[it->first];;
      std::cout<<"IN\n";
      GridFunctionToBlueprintField(lo_gf, n_field);
      std::cout<<"OUT\n";
      // all supported grid functions coming out of mfem end up being associtated with vertices
      n_field["association"] = "vertex";
      
      //delete lo_col;
      //delete lo_fes;
      //delete lo_gf;
    }
    
    conduit::Node info;
    bool success = conduit::blueprint::verify("mesh",n_dset,info);
    if(!success) 
    {
      info.print();
      ASCENT_ERROR("Linearize: failed to build a blueprint conforming data set from mfem") 
    }
    //delete lo_mesh;

  }
  //output.print();
}

void
MFEMDataAdapter::GridFunctionToBlueprintField(mfem::GridFunction *gf,
                                              Node &n_field,
                                              const std::string &main_topology_name)
{
   n_field["basis"] = gf->FESpace()->FEColl()->Name();
   n_field["topology"] = main_topology_name;

   int vdim  = gf->FESpace()->GetVDim();
   int ndofs = gf->FESpace()->GetNDofs();

   if (vdim == 1) // scalar case
   {
      //n_field["values"].set_external(gf->GetData(),
      //                               ndofs);
      n_field["values"].set(gf->GetData(),
                            ndofs);
   }
   else // vector case
   {
      // deal with striding of all components

     mfem::Ordering::Type ordering = gf->FESpace()->GetOrdering();

      int entry_stride = (ordering == mfem::Ordering::byNODES ? 1 : vdim);
      int vdim_stride  = (ordering == mfem::Ordering::byNODES ? ndofs : 1);

      index_t offset = 0;
      index_t stride = sizeof(double) * entry_stride;

      for (int d = 0;  d < vdim; d++)
      {
         std::ostringstream oss;
         oss << "v" << d;
         std::string comp_name = oss.str();
         //n_field["values"][comp_name].set_external(gf->GetData(),
         //                                          ndofs,
         //                                          offset,
         //                                          stride);
         n_field["values"][comp_name].set(gf->GetData(),
                                          ndofs,
                                          offset,
                                          stride);
         offset +=  sizeof(double) * vdim_stride;
      }
   }

}
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
