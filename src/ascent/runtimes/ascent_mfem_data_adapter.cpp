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

//-----------------------------------------------------------------------------
// VTKHDataAdapter public methods
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
      const conduit::Node &dom = node.child(i);      
      // this should exist
      int domain_id = dom["state/domain_id"].to_int();
      // insert domain conversion

      bool zero_copy = false;
      // TODO: tell collection not to delete it
      mfem::Mesh *mesh = nullptr;
      mesh = mfem::ConduitDataCollection::BlueprintMeshToMesh(dom, topo_name, zero_copy);

      //dom.print();
      std::cout<<"MESH "<<mesh->GetNE()<<"\n";
      mfem::ConduitDataCollection *col = new mfem::ConduitDataCollection("", mesh);

      if(node.has_path("state/cycle"))
      {
        int cycle = node["state/cycle"].to_int32();
        col->SetCycle(cycle);
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
            col->RegisterField(fnames[f], gf); 
          }
        }
      }

      res->m_data_sets.push_back(col);
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
    mfem::Mesh *ho_mesh = ho_domains->m_data_sets[i]->GetMesh(); 
    const mfem::FiniteElementSpace *ho_fes_space = ho_mesh->GetNodalFESpace();
    const mfem::FiniteElementCollection *ho_fes_col = ho_fes_space->FEColl();
    // refine the mesh and convert to blueprint
    mfem::Mesh *lo_mesh = new mfem::Mesh(ho_mesh, refinement, mfem::BasisType::GaussLobatto); 
    mfem::ConduitDataCollection::MeshToBlueprintMesh (lo_mesh, n_dset);
    std::ofstream lovtk("low.vtk"); 
    lo_mesh->PrintVTK(lovtk);
    //mfem::ConduitDataCollection::MeshToBlueprintMesh (ho_mesh, n_dset);
    
    int conn_size = n_dset["topologies/main/elements/connectivity"].dtype().number_of_elements();
    std::cout<<"conn size() "<<conn_size<<"\n";
    int *conn = n_dset["topologies/main/elements/connectivity"].as_int32_ptr();

    const mfem::Element *el = lo_mesh->GetElement(0);
    const int *verts = el->GetVertices();
    std::cout<<" ** ";
    for(int x = 0; x < 8; x++)
    {
      std::cout<<verts[x]<<" ";
      for(int y = 0; y < 3; y++) 
      {
        std::cout<<lo_mesh->GetVertex(verts[x])[0]<<" ";
        std::cout<<lo_mesh->GetVertex(verts[x])[1]<<" ";
        std::cout<<lo_mesh->GetVertex(verts[x])[2]<<"\n";
      }
    }
    std::cout<<"\n";

    std::map<int,int> hash;
    for(int x = 0; x < conn_size; x++)
    {
      hash[conn[x]]++;
    }

    int first[8];
    for(int x = 0; x < conn_size; x++)
    {
      if(x < 8) first[x] = conn[x];
      hash[conn[x]]++;
    }

    int max = 0;
    int min = 100;
    int el_id = 0;
    for(auto it = hash.begin(); it != hash.end(); it++)
    {
      if(it->second > max) 
      {
        el_id = it->first;
        max = it->second;
      }
      if(it->second < min) 
      {
        min = it->second;
      }
    }
    std::cout<<"Max count "<<max<<" conn "<<el_id<<"\n";
    std::cout<<"Min count "<<min<<"\n";
    double *cx = n_dset["coordsets/coords/values/x"].as_float64_ptr();
    double *cy = n_dset["coordsets/coords/values/y"].as_float64_ptr();
    double *cz = n_dset["coordsets/coords/values/z"].as_float64_ptr();
    std::cout<<"X0 "<<cx[0]<<"\n";
    std::cout<<"y0 "<<cy[0]<<"\n";
    std::cout<<"z0 "<<cz[0]<<"\n";
    std::ofstream obj("output.obj");
    for(int x = 0; x < 8; x++)
    {
      obj<<"v "<<cx[first[x]]<<" "<<cy[first[x]]<<" "<<cz[first[x]]<<"\n";
      std::cout<<" "<<first[x];
    }
    std::cout<<"\n";
    obj<<"f 1 2 6 5\n";
    obj<<"f 1 2 3 4\n";
    obj<<"f 2 3 7 6\n";
    obj<<"f 5 6 7 8\n";
    obj<<"f 3 4 8 7\n";
    obj<<"f 1 5 8 4\n";
    obj.close();
    //int dims = ho_mesh->Dimension();

    conduit::Node &n_fields = n_dset["fields"];
    auto field_map = ho_domains->m_data_sets[i]->GetFieldMap();

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
      mfem::ConduitDataCollection::GridFunctionToBlueprintField(lo_gf, n_field);
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

};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
